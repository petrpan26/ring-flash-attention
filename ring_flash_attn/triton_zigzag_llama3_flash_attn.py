"""
Fused Triton Implementation of Zigzag Llama3 Flash Attention

This implementation solves the inefficiency of the original Python-based zigzag_llama3
implementation by fusing the two Q group processing into a single Triton kernel.

INEFFICIENCY IN ORIGINAL IMPLEMENTATION:
=========================================
The original execute_grouped_attention() processes Q in 2 groups (early/late chunks):
- Group 0 (early chunks): Loads K,V, calls flash_attn → reads K,V from HBM
- Group 1 (late chunks): Loads K,V, calls flash_attn → reads K,V from HBM again

Total K,V memory reads: 2x (once per group)

OPTIMIZATION WITH FUSED TRITON KERNEL:
======================================
This implementation:
1. Loads each K,V tile once into SRAM
2. Processes BOTH Q groups using the same K,V tile
3. Applies group-specific causal masking based on chunk indices
4. Writes separate outputs for each group

Total K,V memory reads: 1x (50% reduction in memory bandwidth!)

KEY DESIGN:
==========
- Q is split into 2 groups by global chunk index:
  * Group 0: chunks [0, ..., world_size-1] (early)
  * Group 1: chunks [world_size, ..., 2*world_size-1] (late)

- Each rank has 2 chunks:
  * chunk_idx_0 = rank (belongs to group 0)
  * chunk_idx_1 = 2*world_size - 1 - rank (belongs to group 1)

- Causal masking: Q chunk i can only attend to K,V chunks [0, ..., i]
  * Group 0 needs K,V up to chunk_idx_0
  * Group 1 needs K,V up to chunk_idx_1 (usually more)

REFERENCES:
===========
- Triton Flash Attention Tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- Original zigzag_llama3 implementation: zigzag_llama3_flash_attn_varlen.py
"""

import math
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.jit
def _fused_zigzag_fwd_kernel(
    # Input tensors
    Q0, Q1,  # Q for group 0 and group 1 (already split)
    K, V,    # K, V in contiguous format (after rearrangement)
    # Output tensors
    Out0, Out1,  # Outputs for group 0 and group 1
    Lse0, Lse1,  # LSE for group 0 and group 1
    TMP,  # Scratchpad buffer
    # Scalar parameters
    softmax_scale,
    # Sequence lengths and offsets
    cu_seqlens_q0,  # Cumulative sequence lengths for Q group 0
    cu_seqlens_q1,  # Cumulative sequence lengths for Q group 1
    cu_seqlens_k,   # Cumulative sequence lengths for K (GLOBAL coordinates)
    chunk_idx_0: tl.constexpr,  # Global chunk index for group 0
    chunk_idx_1: tl.constexpr,  # Global chunk index for group 1
    total_chunks: tl.constexpr,  # Total chunks (2 * world_size)
    # Strides
    stride_q0b, stride_q0h, stride_q0m,
    stride_q1b, stride_q1h, stride_q1m,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_o0b, stride_o0h, stride_o0m,
    stride_o1b, stride_o1h, stride_o1m,
    # Dimensions
    nheads,
    seqlen_q0,  # Max seqlen for Q group 0
    seqlen_q1,  # Max seqlen for Q group 1
    seqlen_k,   # Max seqlen for K (GLOBAL)
    seqlen_q0_rounded,
    seqlen_q1_rounded,
    headdim,
    # Compile-time constants
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M0: tl.constexpr,
    EVEN_M1: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_VARLEN: tl.constexpr,  # Whether to use variable-length sequences
):
    """
    Fused forward kernel that processes both Q groups in a single pass.

    For each K,V tile loaded into SRAM, we:
    1. Process Q group 0 (early chunks) with causal mask up to chunk_idx_0
    2. Process Q group 1 (late chunks) with causal mask up to chunk_idx_1

    This halves the K,V memory bandwidth compared to separate kernels.
    """
    # Get program IDs
    start_m = tl.program_id(0)  # Q block index
    off_hb = tl.program_id(1)   # Batch * head index
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # Determine which Q group this block processes
    # We alternate between group 0 and group 1 blocks
    # Actually, we need to process BOTH groups for each Q block
    # Let's process group 0 first, then group 1

    # This kernel processes Q group 0
    # We'll launch two separate grids, one for each Q group
    # But we want to FUSE them...

    # Alternative approach: process both groups in the same kernel
    # by loading Q for both groups

    # Let me rethink this...
    # The key insight is that we want to load K,V once and process both Q groups
    # So the kernel should:
    # 1. Load Q block from group 0
    # 2. Load Q block from group 1
    # 3. Loop over K,V blocks:
    #    - Load K,V block
    #    - Process with Q group 0 (if within causal range)
    #    - Process with Q group 1 (if within causal range)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers for Q group 0
    q0_ptrs = (
        Q0 + off_b * stride_q0b + off_h * stride_q0h +
        (offs_m[:, None] * stride_q0m + offs_d[None, :])
    )

    # Initialize pointers for Q group 1
    q1_ptrs = (
        Q1 + off_b * stride_q1b + off_h * stride_q1h +
        (offs_m[:, None] * stride_q1m + offs_d[None, :])
    )

    # Initialize pointers for K, V
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh +
        (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh +
        (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # Initialize accumulators and statistics for both groups
    acc_o0 = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o1 = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    lse_i0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Temporary buffer pointers
    t_ptrs0 = TMP + off_hb * (seqlen_q0_rounded + seqlen_q1_rounded) + offs_m
    t_ptrs1 = TMP + off_hb * (seqlen_q0_rounded + seqlen_q1_rounded) + seqlen_q0_rounded + offs_m

    # Load Q for both groups (will stay in SRAM)
    if EVEN_M0:
        if EVEN_HEADDIM:
            q0 = tl.load(q0_ptrs)
        else:
            q0 = tl.load(q0_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q0 = tl.load(q0_ptrs, mask=offs_m[:, None] < seqlen_q0, other=0.0)
        else:
            q0 = tl.load(
                q0_ptrs,
                mask=(offs_m[:, None] < seqlen_q0) & (offs_d[None, :] < headdim),
                other=0.0
            )

    if EVEN_M1:
        if EVEN_HEADDIM:
            q1 = tl.load(q1_ptrs)
        else:
            q1 = tl.load(q1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q1 = tl.load(q1_ptrs, mask=offs_m[:, None] < seqlen_q1, other=0.0)
        else:
            q1 = tl.load(
                q1_ptrs,
                mask=(offs_m[:, None] < seqlen_q1) & (offs_d[None, :] < headdim),
                other=0.0
            )

    # Compute K,V range for each group based on chunk indices
    # Group 0 needs K,V up to end of chunk_idx_0
    # Group 1 needs K,V up to end of chunk_idx_1
    chunk_size = seqlen_k // total_chunks
    end_n0 = (chunk_idx_0 + 1) * chunk_size
    end_n1 = (chunk_idx_1 + 1) * chunk_size

    # Loop over K,V blocks - we use the max range needed
    end_n_max = tl.maximum(end_n0, end_n1)

    for start_n in range(0, end_n_max, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K, V (shared by both groups)
        if EVEN_N:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        # Process Q group 0 with this K,V block
        if start_n < end_n0:
            qk0 = tl.dot(q0, k, trans_b=True)

            # Apply masks
            if not EVEN_N:
                qk0 = tl.where((start_n + offs_n)[None, :] < end_n0, qk0, float("-inf"))
            if IS_CAUSAL:
                # For zigzag distribution, causal mask is position-based in GLOBAL coordinates
                # Q positions: start_m * BLOCK_M to (start_m + 1) * BLOCK_M
                # K positions: start_n to start_n + BLOCK_N
                # Causal: q_pos >= k_pos in global coordinates
                qk0 = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk0, float("-inf"))

            # Softmax computation
            m_ij0 = tl.maximum(tl.max(qk0, 1) * softmax_scale, lse_i0)
            p0 = tl.exp(qk0 * softmax_scale - m_ij0[:, None])
            l_ij0 = tl.sum(p0, 1)

            # Scale and update accumulator
            acc_o_scale0 = tl.exp(m_i0 - m_ij0)
            tl.store(t_ptrs0, acc_o_scale0)
            acc_o_scale0 = tl.load(t_ptrs0)
            acc_o0 = acc_o0 * acc_o_scale0[:, None]
            acc_o0 += tl.dot(p0.to(v.dtype), v)

            # Update statistics
            m_i0 = m_ij0
            l_i_new0 = tl.exp(lse_i0 - m_ij0) + l_ij0
            lse_i0 = m_ij0 + tl.log(l_i_new0)

        # Process Q group 1 with this K,V block
        if start_n < end_n1:
            qk1 = tl.dot(q1, k, trans_b=True)

            # Apply masks
            if not EVEN_N:
                qk1 = tl.where((start_n + offs_n)[None, :] < end_n1, qk1, float("-inf"))
            if IS_CAUSAL:
                qk1 = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk1, float("-inf"))

            # Softmax computation
            m_ij1 = tl.maximum(tl.max(qk1, 1) * softmax_scale, lse_i1)
            p1 = tl.exp(qk1 * softmax_scale - m_ij1[:, None])
            l_ij1 = tl.sum(p1, 1)

            # Scale and update accumulator
            acc_o_scale1 = tl.exp(m_i1 - m_ij1)
            tl.store(t_ptrs1, acc_o_scale1)
            acc_o_scale1 = tl.load(t_ptrs1)
            acc_o1 = acc_o1 * acc_o_scale1[:, None]
            acc_o1 += tl.dot(p1.to(v.dtype), v)

            # Update statistics
            m_i1 = m_ij1
            l_i_new1 = tl.exp(lse_i1 - m_ij1) + l_ij1
            lse_i1 = m_ij1 + tl.log(l_i_new1)

    # Final scaling and write outputs for group 0
    o_scale0 = tl.exp(m_i0 - lse_i0)
    tl.store(t_ptrs0, o_scale0)
    o_scale0 = tl.load(t_ptrs0)
    acc_o0 = acc_o0 * o_scale0[:, None]

    # Write LSE for group 0
    lse_ptrs0 = Lse0 + off_hb * seqlen_q0_rounded + offs_m
    tl.store(lse_ptrs0, lse_i0)

    # Write output for group 0
    out_ptrs0 = (
        Out0 + off_b * stride_o0b + off_h * stride_o0h +
        (offs_m[:, None] * stride_o0m + offs_d[None, :])
    )
    if EVEN_M0:
        if EVEN_HEADDIM:
            tl.store(out_ptrs0, acc_o0)
        else:
            tl.store(out_ptrs0, acc_o0, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs0, acc_o0, mask=offs_m[:, None] < seqlen_q0)
        else:
            tl.store(
                out_ptrs0, acc_o0,
                mask=(offs_m[:, None] < seqlen_q0) & (offs_d[None, :] < headdim)
            )

    # Final scaling and write outputs for group 1
    o_scale1 = tl.exp(m_i1 - lse_i1)
    tl.store(t_ptrs1, o_scale1)
    o_scale1 = tl.load(t_ptrs1)
    acc_o1 = acc_o1 * o_scale1[:, None]

    # Write LSE for group 1
    lse_ptrs1 = Lse1 + off_hb * seqlen_q1_rounded + offs_m
    tl.store(lse_ptrs1, lse_i1)

    # Write output for group 1
    out_ptrs1 = (
        Out1 + off_b * stride_o1b + off_h * stride_o1h +
        (offs_m[:, None] * stride_o1m + offs_d[None, :])
    )
    if EVEN_M1:
        if EVEN_HEADDIM:
            tl.store(out_ptrs1, acc_o1)
        else:
            tl.store(out_ptrs1, acc_o1, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs1, acc_o1, mask=offs_m[:, None] < seqlen_q1)
        else:
            tl.store(
                out_ptrs1, acc_o1,
                mask=(offs_m[:, None] < seqlen_q1) & (offs_d[None, :] < headdim)
            )


# ============================================================================
# Python Wrapper for Forward Pass
# ============================================================================

def fused_zigzag_llama3_flash_attn_forward(
    q0: torch.Tensor,  # Q for group 0 [tokens, nheads, headdim]
    q1: torch.Tensor,  # Q for group 1 [tokens, nheads, headdim]
    k: torch.Tensor,   # K in contiguous format [total_tokens, nheads, headdim]
    v: torch.Tensor,   # V in contiguous format [total_tokens, nheads, headdim]
    cu_seqlens_q0: torch.Tensor,  # Cumulative seqlens for Q group 0
    cu_seqlens_q1: torch.Tensor,  # Cumulative seqlens for Q group 1
    cu_seqlens_k: torch.Tensor,   # Cumulative seqlens for K (GLOBAL)
    chunk_idx_0: int,  # Global chunk index for group 0
    chunk_idx_1: int,  # Global chunk index for group 1
    total_chunks: int,  # Total number of chunks (2 * world_size)
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused forward pass for zigzag llama3 flash attention.

    Processes both Q groups in a single kernel, loading K,V only once.

    Args:
        q0, q1: Query tensors for group 0 and 1 (already split by chunks)
        k, v: Key, Value tensors in contiguous format (after rearrangement)
        cu_seqlens_q0, cu_seqlens_q1: Cumulative sequence lengths for Q groups
        cu_seqlens_k: Cumulative sequence lengths for K (GLOBAL coordinates)
        chunk_idx_0, chunk_idx_1: Global chunk indices for the two groups
        total_chunks: Total number of chunks (2 * world_size)
        softmax_scale: Scaling factor for softmax
        causal: Whether to use causal attention

    Returns:
        out0, out1: Output tensors for group 0 and 1
        lse0, lse1: Log-sum-exp values for group 0 and 1
    """
    # Validate inputs
    assert q0.dim() == 3 and q1.dim() == 3
    assert k.dim() == 3 and v.dim() == 3
    assert q0.shape[-1] == q1.shape[-1] == k.shape[-1] == v.shape[-1]

    batch_size = 1  # For varlen, batch is always 1
    nheads = q0.shape[1]
    headdim = q0.shape[2]
    seqlen_q0 = q0.shape[0]
    seqlen_q1 = q1.shape[0]
    seqlen_k = k.shape[0]

    softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))

    # Allocate outputs
    out0 = torch.empty_like(q0)
    out1 = torch.empty_like(q1)

    # Round up sequence lengths
    seqlen_q0_rounded = math.ceil(seqlen_q0 / 128) * 128
    seqlen_q1_rounded = math.ceil(seqlen_q1 / 128) * 128

    # Allocate LSE and temporary buffers
    lse0 = torch.empty((batch_size, nheads, seqlen_q0_rounded), device=q0.device, dtype=torch.float32)
    lse1 = torch.empty((batch_size, nheads, seqlen_q1_rounded), device=q1.device, dtype=torch.float32)
    tmp = torch.empty(
        (batch_size, nheads, seqlen_q0_rounded + seqlen_q1_rounded),
        device=q0.device, dtype=torch.float32
    )

    # Compute block sizes
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    BLOCK_M = 128
    BLOCK_N = 128
    num_warps = 4 if headdim <= 64 else 8

    # Launch kernel - process both groups in the same grid
    # We need max(seqlen_q0, seqlen_q1) blocks
    max_seqlen_q = max(seqlen_q0, seqlen_q1)
    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size * nheads)

    _fused_zigzag_fwd_kernel[grid](
        q0, q1, k, v,
        out0, out1,
        lse0, lse1,
        tmp,
        softmax_scale,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        chunk_idx_0, chunk_idx_1, total_chunks,
        # Strides for q0
        0, 0, q0.stride(0),  # batch=1, so batch stride is 0
        # Strides for q1
        0, 0, q1.stride(0),
        # Strides for k, v
        0, 0, k.stride(0),
        0, 0, v.stride(0),
        # Strides for out0, out1
        0, 0, out0.stride(0),
        0, 0, out1.stride(0),
        # Dimensions
        nheads,
        seqlen_q0, seqlen_q1, seqlen_k,
        seqlen_q0_rounded, seqlen_q1_rounded,
        headdim,
        # Compile-time constants
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M0=seqlen_q0 % BLOCK_M == 0,
        EVEN_M1=seqlen_q1 % BLOCK_M == 0,
        EVEN_N=seqlen_k % BLOCK_N == 0,
        EVEN_HEADDIM=headdim == BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_VARLEN=True,
        num_warps=num_warps,
        num_stages=1,
    )

    return out0, out1, lse0, lse1


# ============================================================================
# Backward Kernels
# ============================================================================

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out, DO, Delta,
    stride_ob, stride_oh, stride_om,
    stride_dob, stride_doh, stride_dom,
    nheads, seqlen_q, seqlen_q_rounded, headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """Preprocess for backward: compute delta = sum(out * dout)"""
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Load out and dout
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _fused_zigzag_bwd_kernel(
    # Forward inputs
    Q0, Q1, K, V,
    # Forward outputs
    Out0, Out1,
    # Backward inputs
    DO0, DO1,
    # LSE and Delta from forward/preprocessing
    Lse0, Lse1, D0, D1,
    # Gradient outputs
    DQ0, DQ1, DK, DV,
    # Scalar parameters
    softmax_scale,
    # Sequence lengths and indices
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    chunk_idx_0: tl.constexpr,
    chunk_idx_1: tl.constexpr,
    total_chunks: tl.constexpr,
    # Strides
    stride_q0m, stride_q1m, stride_kn, stride_vn,
    stride_o0m, stride_o1m,
    stride_do0m, stride_do1m,
    stride_dq0m, stride_dq1m, stride_dkn, stride_dvn,
    # Dimensions
    seqlen_q0, seqlen_q1, seqlen_k,
    seqlen_q0_rounded, seqlen_q1_rounded,
    headdim,
    # Compile-time constants
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M0: tl.constexpr,
    EVEN_M1: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused backward kernel that processes both Q groups in a single pass.

    For each K,V tile loaded, we:
    1. Compute dK, dV contributions from Q group 0
    2. Compute dK, dV contributions from Q group 1
    3. Accumulate both into dK, dV (since they may overlap)

    This halves the K,V memory bandwidth compared to separate kernels.
    """
    # This kernel processes one K,V block
    start_n = tl.program_id(0) * BLOCK_N
    off_h = tl.program_id(1)

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers for K, V
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])

    # Load K, V (will stay in SRAM)
    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0
            )

    # Initialize gradient accumulators for K, V
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # Compute K,V range for each group
    chunk_size = seqlen_k // total_chunks
    end_n0 = (chunk_idx_0 + 1) * chunk_size
    end_n1 = (chunk_idx_1 + 1) * chunk_size

    # Check if this K,V block is needed by each group
    block_needed_by_group0 = start_n < end_n0
    block_needed_by_group1 = start_n < end_n1

    # Process Q group 0 if this K,V block is in range
    if block_needed_by_group0:
        # Determine which Q blocks from group 0 can attend to this K,V block
        # For causal attention: Q position >= K position
        # Q positions: 0 to seqlen_q0
        # K positions: start_n to start_n + BLOCK_N
        # Begin from the Q block that first includes positions >= start_n
        if IS_CAUSAL:
            begin_m0 = (start_n // BLOCK_M) * BLOCK_M
        else:
            begin_m0 = 0

        num_block_m0 = tl.cdiv(seqlen_q0, BLOCK_M)

        # Loop over Q blocks for group 0
        for block_m_idx in range(begin_m0 // BLOCK_M, num_block_m0):
            start_m0 = block_m_idx * BLOCK_M
            offs_m_curr = start_m0 + offs_m

            # Pointers for this Q block
            q0_ptrs = Q0 + (offs_m_curr[:, None] * stride_q0m + offs_d[None, :])
            do0_ptrs = DO0 + (offs_m_curr[:, None] * stride_do0m + offs_d[None, :])
            dq0_ptrs = DQ0 + (offs_m_curr[:, None] * stride_dq0m + offs_d[None, :])

            # Load Q, DO for group 0
            if EVEN_M0 and EVEN_HEADDIM:
                q0 = tl.load(q0_ptrs)
                do0 = tl.load(do0_ptrs)
            else:
                if EVEN_HEADDIM:
                    q0 = tl.load(q0_ptrs, mask=offs_m_curr[:, None] < seqlen_q0, other=0.0)
                    do0 = tl.load(do0_ptrs, mask=offs_m_curr[:, None] < seqlen_q0, other=0.0)
                else:
                    q0 = tl.load(
                        q0_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q0) & (offs_d[None, :] < headdim),
                        other=0.0
                    )
                    do0 = tl.load(
                        do0_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q0) & (offs_d[None, :] < headdim),
                        other=0.0
                    )

            # Recompute attention scores
            qk0 = tl.dot(q0, k, trans_b=True)

            # Apply masks (same as forward)
            if not EVEN_N:
                qk0 = tl.where(offs_n[None, :] < end_n0, qk0, float("-inf"))
            if IS_CAUSAL:
                qk0 = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk0, float("-inf"))

            # Load LSE and compute softmax
            lse_i0 = tl.load(Lse0 + off_h * seqlen_q0_rounded + offs_m_curr)
            p0 = tl.exp(qk0 * softmax_scale - lse_i0[:, None])

            # Compute dV contribution from this Q block
            dv += tl.dot(p0.to(do0.dtype), do0, trans_a=True)

            # Compute dP = dO @ V^T
            dp0 = tl.dot(do0, v, trans_b=True)

            # Load delta for this Q block
            Di0 = tl.load(D0 + off_h * seqlen_q0_rounded + offs_m_curr)

            # Compute dS = P * (dP - delta)
            ds0 = (p0 * (dp0 - Di0[:, None]) * softmax_scale).to(q0.dtype)

            # Compute dK contribution from this Q block
            dk += tl.dot(ds0, q0, trans_a=True)

            # Compute dQ for this Q block
            dq0 = tl.dot(ds0, k)

            # Store dQ (accumulate if multiple K,V blocks contribute)
            if EVEN_M0 and EVEN_HEADDIM:
                dq0_existing = tl.load(dq0_ptrs)
                tl.store(dq0_ptrs, dq0_existing + dq0)
            else:
                if EVEN_HEADDIM:
                    dq0_existing = tl.load(dq0_ptrs, mask=offs_m_curr[:, None] < seqlen_q0, other=0.0)
                    tl.store(dq0_ptrs, dq0_existing + dq0, mask=offs_m_curr[:, None] < seqlen_q0)
                else:
                    dq0_existing = tl.load(
                        dq0_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q0) & (offs_d[None, :] < headdim),
                        other=0.0
                    )
                    tl.store(
                        dq0_ptrs, dq0_existing + dq0,
                        mask=(offs_m_curr[:, None] < seqlen_q0) & (offs_d[None, :] < headdim)
                    )

    # Process Q group 1 (similar to group 0)
    if block_needed_by_group1:
        if IS_CAUSAL:
            begin_m1 = (start_n // BLOCK_M) * BLOCK_M
        else:
            begin_m1 = 0

        num_block_m1 = tl.cdiv(seqlen_q1, BLOCK_M)

        for block_m_idx in range(begin_m1 // BLOCK_M, num_block_m1):
            start_m1 = block_m_idx * BLOCK_M
            offs_m_curr = start_m1 + offs_m

            q1_ptrs = Q1 + (offs_m_curr[:, None] * stride_q1m + offs_d[None, :])
            do1_ptrs = DO1 + (offs_m_curr[:, None] * stride_do1m + offs_d[None, :])
            dq1_ptrs = DQ1 + (offs_m_curr[:, None] * stride_dq1m + offs_d[None, :])

            if EVEN_M1 and EVEN_HEADDIM:
                q1 = tl.load(q1_ptrs)
                do1 = tl.load(do1_ptrs)
            else:
                if EVEN_HEADDIM:
                    q1 = tl.load(q1_ptrs, mask=offs_m_curr[:, None] < seqlen_q1, other=0.0)
                    do1 = tl.load(do1_ptrs, mask=offs_m_curr[:, None] < seqlen_q1, other=0.0)
                else:
                    q1 = tl.load(
                        q1_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q1) & (offs_d[None, :] < headdim),
                        other=0.0
                    )
                    do1 = tl.load(
                        do1_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q1) & (offs_d[None, :] < headdim),
                        other=0.0
                    )

            qk1 = tl.dot(q1, k, trans_b=True)

            if not EVEN_N:
                qk1 = tl.where(offs_n[None, :] < end_n1, qk1, float("-inf"))
            if IS_CAUSAL:
                qk1 = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk1, float("-inf"))

            lse_i1 = tl.load(Lse1 + off_h * seqlen_q1_rounded + offs_m_curr)
            p1 = tl.exp(qk1 * softmax_scale - lse_i1[:, None])

            dv += tl.dot(p1.to(do1.dtype), do1, trans_a=True)

            dp1 = tl.dot(do1, v, trans_b=True)
            Di1 = tl.load(D1 + off_h * seqlen_q1_rounded + offs_m_curr)
            ds1 = (p1 * (dp1 - Di1[:, None]) * softmax_scale).to(q1.dtype)

            dk += tl.dot(ds1, q1, trans_a=True)

            dq1 = tl.dot(ds1, k)

            if EVEN_M1 and EVEN_HEADDIM:
                dq1_existing = tl.load(dq1_ptrs)
                tl.store(dq1_ptrs, dq1_existing + dq1)
            else:
                if EVEN_HEADDIM:
                    dq1_existing = tl.load(dq1_ptrs, mask=offs_m_curr[:, None] < seqlen_q1, other=0.0)
                    tl.store(dq1_ptrs, dq1_existing + dq1, mask=offs_m_curr[:, None] < seqlen_q1)
                else:
                    dq1_existing = tl.load(
                        dq1_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q1) & (offs_d[None, :] < headdim),
                        other=0.0
                    )
                    tl.store(
                        dq1_ptrs, dq1_existing + dq1,
                        mask=(offs_m_curr[:, None] < seqlen_q1) & (offs_d[None, :] < headdim)
                    )

    # Write back dK, dV for this K,V block
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])

    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk)
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


# ============================================================================
# Python Wrapper for Backward Pass
# ============================================================================

def fused_zigzag_llama3_flash_attn_backward(
    do0: torch.Tensor,
    do1: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    lse0: torch.Tensor,
    lse1: torch.Tensor,
    cu_seqlens_q0: torch.Tensor,
    cu_seqlens_q1: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    chunk_idx_0: int,
    chunk_idx_1: int,
    total_chunks: int,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused backward pass for zigzag llama3 flash attention.

    Returns:
        dq0, dq1, dk, dv: Gradients for Q groups 0 and 1, and K, V
    """
    batch_size = 1
    nheads = q0.shape[1]
    headdim = q0.shape[2]
    seqlen_q0 = q0.shape[0]
    seqlen_q1 = q1.shape[0]
    seqlen_k = k.shape[0]

    seqlen_q0_rounded = math.ceil(seqlen_q0 / 128) * 128
    seqlen_q1_rounded = math.ceil(seqlen_q1 / 128) * 128

    # Allocate gradient tensors
    dq0 = torch.zeros_like(q0)
    dq1 = torch.zeros_like(q1)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Allocate delta buffers
    delta0 = torch.empty((batch_size, nheads, seqlen_q0_rounded), device=q0.device, dtype=torch.float32)
    delta1 = torch.empty((batch_size, nheads, seqlen_q1_rounded), device=q1.device, dtype=torch.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    BLOCK_M = 128
    BLOCK_N = 128

    # Preprocess: compute delta0 and delta1
    grid = lambda META: (triton.cdiv(seqlen_q0, META["BLOCK_M"]), batch_size * nheads)
    _bwd_preprocess_do_o_dot[grid](
        out0, do0, delta0,
        0, 0, out0.stride(0),
        0, 0, do0.stride(0),
        nheads, seqlen_q0, seqlen_q0_rounded, headdim,
        BLOCK_M=BLOCK_M,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    grid = lambda META: (triton.cdiv(seqlen_q1, META["BLOCK_M"]), batch_size * nheads)
    _bwd_preprocess_do_o_dot[grid](
        out1, do1, delta1,
        0, 0, out1.stride(0),
        0, 0, do1.stride(0),
        nheads, seqlen_q1, seqlen_q1_rounded, headdim,
        BLOCK_M=BLOCK_M,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # Main backward kernel - process K,V blocks
    grid = (triton.cdiv(seqlen_k, BLOCK_N), batch_size * nheads)
    num_warps = 4 if headdim <= 64 else 8

    _fused_zigzag_bwd_kernel[grid](
        q0, q1, k, v,
        out0, out1,
        do0, do1,
        lse0, lse1, delta0, delta1,
        dq0, dq1, dk, dv,
        softmax_scale,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        chunk_idx_0, chunk_idx_1, total_chunks,
        # Strides (batch=1, so many strides are 0)
        q0.stride(0), q1.stride(0), k.stride(0), v.stride(0),
        out0.stride(0), out1.stride(0),
        do0.stride(0), do1.stride(0),
        dq0.stride(0), dq1.stride(0), dk.stride(0), dv.stride(0),
        # Dimensions
        seqlen_q0, seqlen_q1, seqlen_k,
        seqlen_q0_rounded, seqlen_q1_rounded,
        headdim,
        # Compile-time constants
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M0=seqlen_q0 % BLOCK_M == 0,
        EVEN_M1=seqlen_q1 % BLOCK_M == 0,
        EVEN_N=seqlen_k % BLOCK_N == 0,
        EVEN_HEADDIM=headdim == BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )

    return dq0, dq1, dk, dv


# ============================================================================
# Integration with existing zigzag_llama3 code
# ============================================================================

def replace_grouped_attention_with_fused_triton(
    chunk_q_list,
    chunk_cu_seqlens_q_list,
    chunk_indices_list,
    kv_buffer,
    kv_slices,
    nheads,
    head_dim,
    softmax_scale,
    dropout_p,
    causal,
    window_size,
    alibi_slopes,
    deterministic,
    world_size,
    cu_seqlens_k,
    chunk_idx_0,
    chunk_idx_1,
):
    """
    Drop-in replacement for execute_grouped_attention() that uses fused Triton kernel.

    This function:
    1. Rearranges KV from zigzag to contiguous (same as before)
    2. Calls fused Triton kernel instead of separate flash_attn calls
    3. Reconstructs outputs in original Q order

    Returns same outputs as execute_grouped_attention for compatibility.
    """
    from .zigzag_llama3_flash_attn_varlen import rearrange_kv_from_zigzag_to_contiguous

    # Rearrange KV to contiguous format (same as original)
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
    k_contiguous = kv_contiguous[0]  # [total_tokens, nheads, headdim]
    v_contiguous = kv_contiguous[1]

    # Assume n_chunks=2, so we have exactly 2 groups
    assert len(chunk_q_list) == 2, "Fused kernel only supports n_chunks=2"

    q0 = chunk_q_list[0]  # Group 0 (early chunks)
    q1 = chunk_q_list[1]  # Group 1 (late chunks)
    cu_seqlens_q0 = chunk_cu_seqlens_q_list[0]
    cu_seqlens_q1 = chunk_cu_seqlens_q_list[1]

    # Build GLOBAL cu_seqlens_k
    # cu_seqlens_k is LOCAL, need to convert to GLOBAL for the kernel
    cu_seqlens_k_global = cu_seqlens_k * world_size

    # Call fused Triton kernel
    out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_forward(
        q0, q1,
        k_contiguous, v_contiguous,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k_global,
        chunk_idx_0, chunk_idx_1,
        total_chunks=2 * world_size,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    # Reconstruct outputs in original Q order (same as original)
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros(
        (total_q, nheads, head_dim),
        dtype=out0.dtype,
        device=out0.device
    )

    # Scatter outputs back to original positions
    indices0 = chunk_indices_list[0]
    indices1 = chunk_indices_list[1]
    out[indices0] = out0
    out[indices1] = out1

    # Reconstruct LSE (handle both possible dimensions)
    if lse0.dim() == 3:
        # LSE is [batch, nheads, seqlen]
        lse = torch.zeros(
            (1, nheads, total_q),
            dtype=lse0.dtype,
            device=lse0.device
        )
        lse[0, :, indices0] = lse0[0, :, :len(indices0)]
        lse[0, :, indices1] = lse1[0, :, :len(indices1)]
        lse = lse[0]  # Remove batch dim
    else:
        # Assume [nheads, seqlen]
        lse = torch.zeros(
            (nheads, total_q),
            dtype=lse0.dtype,
            device=lse0.device
        )
        lse[:, indices0] = lse0[:, :len(indices0)]
        lse[:, indices1] = lse1[:, :len(indices1)]

    # Return chunk_info for backward compatibility
    chunk_info = {
        'chunk_indices_list': chunk_indices_list,
        'chunk_cu_seqlens_q_list': chunk_cu_seqlens_q_list,
        'kv_slices': kv_slices,
    }

    return out, lse, chunk_info
