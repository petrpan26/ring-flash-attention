"""
Fused Triton Kernel for Dual-Group Varlen Flash Attention

This kernel fuses two Q groups (Q0, Q1) into a single kernel, reducing K,V memory reads
from 2x to 1x, achieving 50% memory bandwidth reduction.

Key features:
- Varlen support: Handles variable-length sequences via cu_seqlens
- Single K,V load: Load each K,V block once, use for both Q0 and Q1
- Different attention ranges: Q0 and Q1 attend to different K,V ranges
- FlagAttention optimizations: log2e/exp2, cache modifiers, Dot I trick
"""

import math
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional, List


# ============================================================================
# Hardware-Specific Configuration
# ============================================================================

def get_fused_fwd_config(B, H, M0, M1, N, D, causal):
    """
    Get optimal configuration for fused forward kernel based on hardware.

    Args:
        B: Batch size
        H: Number of heads
        M0, M1: Max sequence lengths for Q group 0 and 1
        N: Max sequence length for K,V
        D: Head dimension
        causal: Whether using causal attention
    """
    device_cap = torch.cuda.get_device_capability()

    if device_cap[0] >= 9:  # H100 or newer
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
    elif device_cap == (8, 0):  # A100
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:  # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
    elif device_cap == (8, 6):  # RTX-3090
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else:  # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:  # Default fallback
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4

    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


# ============================================================================
# Fused Dual-Group Forward Kernel with Varlen Support
# ============================================================================

@triton.jit
def _fused_dual_group_fwd_kernel_varlen(
    # Q groups
    Q0, Q1,
    # Shared K,V
    K, V,
    # Outputs
    Out0, Out1, Lse0, Lse1, TMP,
    # Scalar params
    sm_scale,
    # Varlen sequence info
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_kv_len_q0, max_kv_len_q1,
    # Strides (varlen: no batch dimension)
    stride_q0h, stride_q0m,
    stride_q1h, stride_q1m,
    stride_kh, stride_kn,
    stride_vh, stride_vn,
    stride_o0h, stride_o0m,
    stride_o1h, stride_o1m,
    # Dimensions
    nheads, nheads_k,
    headdim,
    # Compile-time constants
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused forward kernel with varlen support.

    Key innovation: Load K,V once, use for both Q0 and Q1.

    This kernel processes one M-block of queries for both Q0 and Q1 groups,
    loading each K,V block once and reusing it for both groups.
    """
    input_dtype = Q0.dtype.element_ty

    # FlagAttention optimization: use log2e and exp2 for better performance
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # Get program IDs
    start_m = tl.program_id(0)  # M-block index
    off_hz = tl.program_id(1)   # head * sequence
    off_z = off_hz // nheads    # sequence index
    off_h = off_hz % nheads     # head index

    # GQA support: compute K,V head index
    num_groups = nheads // nheads_k
    off_hk = off_h // num_groups

    # Varlen: Load sequence boundaries for this sequence
    seq_start_q0 = tl.load(cu_seqlens_q0 + off_z)
    seq_end_q0 = tl.load(cu_seqlens_q0 + off_z + 1)
    seq_len_q0 = seq_end_q0 - seq_start_q0

    seq_start_q1 = tl.load(cu_seqlens_q1 + off_z)
    seq_end_q1 = tl.load(cu_seqlens_q1 + off_z + 1)
    seq_len_q1 = seq_end_q1 - seq_start_q1

    seq_start_k = tl.load(cu_seqlens_k + off_z)
    seq_end_k = tl.load(cu_seqlens_k + off_z + 1)
    seq_len_k = seq_end_k - seq_start_k

    # Compute attention ranges for this sequence
    end_n0 = tl.minimum(max_kv_len_q0, seq_len_k)
    end_n1 = tl.minimum(max_kv_len_q1, seq_len_k)
    max_end_n = tl.maximum(end_n0, end_n1)

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q0, Q1 (adjusted to this sequence)
    q0_ptrs = (
        Q0 + seq_start_q0 * stride_q0m + off_h * stride_q0h +
        (offs_m[:, None] * stride_q0m + offs_d[None, :])
    )
    q1_ptrs = (
        Q1 + seq_start_q1 * stride_q1m + off_h * stride_q1h +
        (offs_m[:, None] * stride_q1m + offs_d[None, :])
    )

    # Initialize pointers to K,V (adjusted to this sequence, use off_hk for GQA)
    k_ptrs = (
        K + seq_start_k * stride_kn + off_hk * stride_kh +
        (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + seq_start_k * stride_vn + off_hk * stride_vh +
        (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # Load Q for both groups
    if EVEN_HEADDIM:
        q0 = tl.load(q0_ptrs, mask=offs_m[:, None] < seq_len_q0, other=0.0, cache_modifier=".cg")
        q1 = tl.load(q1_ptrs, mask=offs_m[:, None] < seq_len_q1, other=0.0, cache_modifier=".cg")
    else:
        q0 = tl.load(
            q0_ptrs,
            mask=(offs_m[:, None] < seq_len_q0) & (offs_d[None, :] < headdim),
            other=0.0,
            cache_modifier=".cg"
        )
        q1 = tl.load(
            q1_ptrs,
            mask=(offs_m[:, None] < seq_len_q1) & (offs_d[None, :] < headdim),
            other=0.0,
            cache_modifier=".cg"
        )

    # FlagAttention optimization: Dot I trick for headdim < 128
    # This forces Q to stay in registers rather than spilling to shared memory
    if BLOCK_HEADDIM < 128:
        I = tl.where(offs_d[:, None] == offs_d,
                     tl.full((BLOCK_HEADDIM, BLOCK_HEADDIM), 1.0, dtype=input_dtype),
                     tl.full((BLOCK_HEADDIM, BLOCK_HEADDIM), 0.0, dtype=input_dtype))
        q0 = tl.dot(q0, I).to(input_dtype)
        q1 = tl.dot(q1, I).to(input_dtype)

    # Initialize accumulators for both groups
    acc_o0 = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o1 = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    m_i0 = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    m_i1 = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    l_i1 = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Temporary buffer pointers (for workaround to force materialization)
    t_ptrs0 = TMP + off_hz * BLOCK_M * 2 + offs_m
    t_ptrs1 = TMP + off_hz * BLOCK_M * 2 + BLOCK_M + offs_m

    # Loop over K,V blocks ONCE
    for start_n in range(0, max_end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        # Load K,V ONCE (FlagAttention: use cache modifiers)
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=offs_n_curr[:, None] < seq_len_k,
                other=0.0,
                cache_modifier=".cg"
            )
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=offs_n_curr[:, None] < seq_len_k,
                other=0.0,
                cache_modifier=".cg"
            )
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=(offs_n_curr[:, None] < seq_len_k) & (offs_d[None, :] < headdim),
                other=0.0,
                cache_modifier=".cg"
            )
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=(offs_n_curr[:, None] < seq_len_k) & (offs_d[None, :] < headdim),
                other=0.0,
                cache_modifier=".cg"
            )

        # Process Q group 0 (if within range)
        if start_n < end_n0:
            # Compute Q0 @ K^T
            s0 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            s0 += tl.dot(q0, k, trans_b=True)

            # Apply causal mask and range mask
            # For Q0: mask out positions beyond end_n0
            range_mask0 = offs_n_curr[None, :] < end_n0
            s0 = tl.where(range_mask0, s0, float("-inf"))

            if IS_CAUSAL:
                # Causal mask: Q position offs_m can only attend to K positions <= offs_m
                causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
                s0 = tl.where(causal_mask, s0, float("-inf"))

            # Apply qk_scale (log2e included)
            s0 = s0 * qk_scale

            # Online softmax with exp2
            m_i_new0 = tl.maximum(m_i0, tl.max(s0, 1))
            alpha0 = tl.math.exp2(m_i0 - m_i_new0)
            p0 = tl.math.exp2(s0 - m_i_new0[:, None])

            # Scale and update accumulator
            # BUG workaround: store and load to force materialization
            tl.store(t_ptrs0, alpha0)
            alpha0 = tl.load(t_ptrs0)
            acc_o0 *= alpha0[:, None]
            acc_o0 += tl.dot(p0.to(input_dtype), v)

            # Update statistics
            l_i0 = l_i0 * alpha0 + tl.sum(p0, 1)
            m_i0 = m_i_new0

        # Process Q group 1 (if within range) - REUSING same k, v!
        if start_n < end_n1:
            # Compute Q1 @ K^T
            s1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            s1 += tl.dot(q1, k, trans_b=True)

            # Apply causal mask and range mask
            # For Q1: mask out positions beyond end_n1
            range_mask1 = offs_n_curr[None, :] < end_n1
            s1 = tl.where(range_mask1, s1, float("-inf"))

            if IS_CAUSAL:
                # Causal mask: Q position offs_m can only attend to K positions <= offs_m
                causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
                s1 = tl.where(causal_mask, s1, float("-inf"))

            # Apply qk_scale (log2e included)
            s1 = s1 * qk_scale

            # Online softmax with exp2
            m_i_new1 = tl.maximum(m_i1, tl.max(s1, 1))
            alpha1 = tl.math.exp2(m_i1 - m_i_new1)
            p1 = tl.math.exp2(s1 - m_i_new1[:, None])

            # Scale and update accumulator
            tl.store(t_ptrs1, alpha1)
            alpha1 = tl.load(t_ptrs1)
            acc_o1 *= alpha1[:, None]
            acc_o1 += tl.dot(p1.to(input_dtype), v)  # REUSING same v!

            # Update statistics
            l_i1 = l_i1 * alpha1 + tl.sum(p1, 1)
            m_i1 = m_i_new1

    # Final scaling and write outputs for group 0
    acc_o0 *= (1.0 / l_i0[:, None])
    lse0 = m_i0 / log2e + tl.log(l_i0)

    # Write output for group 0
    out_ptrs0 = (
        Out0 + seq_start_q0 * stride_o0m + off_h * stride_o0h +
        (offs_m[:, None] * stride_o0m + offs_d[None, :])
    )
    if EVEN_HEADDIM:
        tl.store(
            out_ptrs0,
            acc_o0.to(input_dtype),
            mask=offs_m[:, None] < seq_len_q0,
            cache_modifier=".cg"
        )
    else:
        tl.store(
            out_ptrs0,
            acc_o0.to(input_dtype),
            mask=(offs_m[:, None] < seq_len_q0) & (offs_d[None, :] < headdim),
            cache_modifier=".cg"
        )

    # Write LSE for group 0
    lse_ptrs0 = Lse0 + off_hz * BLOCK_M + offs_m
    tl.store(lse_ptrs0, lse0, mask=offs_m < seq_len_q0, cache_modifier=".cg")

    # Final scaling and write outputs for group 1
    acc_o1 *= (1.0 / l_i1[:, None])
    lse1 = m_i1 / log2e + tl.log(l_i1)

    # Write output for group 1
    out_ptrs1 = (
        Out1 + seq_start_q1 * stride_o1m + off_h * stride_o1h +
        (offs_m[:, None] * stride_o1m + offs_d[None, :])
    )
    if EVEN_HEADDIM:
        tl.store(
            out_ptrs1,
            acc_o1.to(input_dtype),
            mask=offs_m[:, None] < seq_len_q1,
            cache_modifier=".cg"
        )
    else:
        tl.store(
            out_ptrs1,
            acc_o1.to(input_dtype),
            mask=(offs_m[:, None] < seq_len_q1) & (offs_d[None, :] < headdim),
            cache_modifier=".cg"
        )

    # Write LSE for group 1
    lse_ptrs1 = Lse1 + off_hz * BLOCK_M + offs_m
    tl.store(lse_ptrs1, lse1, mask=offs_m < seq_len_q1, cache_modifier=".cg")


# ============================================================================
# Python Wrapper for Fused Forward Pass
# ============================================================================

def fused_zigzag_llama3_flash_attn_varlen_forward(
    q0: torch.Tensor,
    q1: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q0: torch.Tensor,
    cu_seqlens_q1: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q0: int,
    max_seqlen_q1: int,
    max_seqlen_k: int,
    max_kv_len_q0: int,
    max_kv_len_q1: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused varlen forward pass with FlagAttention optimizations.

    This function fuses two Q groups into a single kernel, reducing K,V memory
    bandwidth by 50% (from 2x to 1x reads).

    Args:
        q0: [total_q0_tokens, nheads, headdim] - Query group 0
        q1: [total_q1_tokens, nheads, headdim] - Query group 1
        k: [total_k_tokens, nheads_k, headdim] - Keys (shared by both groups)
        v: [total_v_tokens, nheads_k, headdim] - Values (shared by both groups)
        cu_seqlens_q0: [batch_size + 1] - Cumulative sequence lengths for Q0
        cu_seqlens_q1: [batch_size + 1] - Cumulative sequence lengths for Q1
        cu_seqlens_k: [batch_size + 1] - Cumulative sequence lengths for K,V
        max_seqlen_q0: Maximum sequence length in Q0
        max_seqlen_q1: Maximum sequence length in Q1
        max_seqlen_k: Maximum sequence length in K,V
        max_kv_len_q0: Max K,V length Q0 can attend to (per sequence)
        max_kv_len_q1: Max K,V length Q1 can attend to (per sequence)
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking

    Returns:
        out0: [total_q0_tokens, nheads, headdim] - Output for Q0
        out1: [total_q1_tokens, nheads, headdim] - Output for Q1
        lse0: [batch_size * nheads, max_seqlen_q0_rounded] - LSE for Q0
        lse1: [batch_size * nheads, max_seqlen_q1_rounded] - LSE for Q1
    """
    # Validate inputs
    batch_size = len(cu_seqlens_q0) - 1
    assert len(cu_seqlens_q1) == batch_size + 1, "Q0 and Q1 must have same batch size"
    assert len(cu_seqlens_k) == batch_size + 1, "K must have same batch size as Q"

    nheads = q0.shape[1]
    nheads_k = k.shape[1]
    headdim = q0.shape[2]

    assert q1.shape[1] == nheads, "Q0 and Q1 must have same number of heads"
    assert q1.shape[2] == headdim, "Q0 and Q1 must have same head dimension"
    assert k.shape[2] == headdim, "K must have same head dimension as Q"
    assert v.shape[2] == headdim, "V must have same head dimension as Q"
    assert nheads % nheads_k == 0, "nheads must be divisible by nheads_k (GQA)"

    softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))

    # Allocate outputs
    out0 = torch.empty_like(q0)
    out1 = torch.empty_like(q1)

    # Allocate LSE tensors
    # Round up to nearest BLOCK_M for alignment
    max_seqlen_q0_rounded = triton.next_power_of_2(max_seqlen_q0)
    max_seqlen_q1_rounded = triton.next_power_of_2(max_seqlen_q1)

    lse0 = torch.empty(
        (batch_size * nheads, max_seqlen_q0_rounded),
        device=q0.device,
        dtype=torch.float32
    )
    lse1 = torch.empty(
        (batch_size * nheads, max_seqlen_q1_rounded),
        device=q1.device,
        dtype=torch.float32
    )

    # Temporary buffer for workaround
    tmp = torch.empty(
        (batch_size * nheads * max(max_seqlen_q0_rounded, max_seqlen_q1_rounded) * 2,),
        device=q0.device,
        dtype=torch.float32
    )

    # Get hardware-specific configuration
    config = get_fused_fwd_config(
        batch_size, nheads, max_seqlen_q0, max_seqlen_q1, max_seqlen_k, headdim, causal
    )
    BLOCK_M, BLOCK_N, num_stages, num_warps = config
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)

    # Launch kernel
    max_seqlen_q = max(max_seqlen_q0, max_seqlen_q1)
    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size * nheads)

    _fused_dual_group_fwd_kernel_varlen[grid](
        # Q groups
        q0, q1,
        # Shared K,V
        k, v,
        # Outputs
        out0, out1, lse0, lse1, tmp,
        # Scalar params
        softmax_scale,
        # Varlen sequence info
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        max_kv_len_q0, max_kv_len_q1,
        # Strides
        q0.stride(1), q0.stride(0),
        q1.stride(1), q1.stride(0),
        k.stride(1), k.stride(0),
        v.stride(1), v.stride(0),
        out0.stride(1), out0.stride(0),
        out1.stride(1), out1.stride(0),
        # Dimensions
        nheads, nheads_k,
        headdim,
        # Compile-time constants
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_HEADDIM=headdim == BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out0, out1, lse0, lse1


# ============================================================================
# Integration Function (Drop-in Replacement)
# ============================================================================

def replace_execute_grouped_attention_with_fused_varlen(
    chunk_q_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    chunk_indices_list: List[torch.Tensor],
    kv_buffer: torch.Tensor,
    kv_slices: List[Tuple[List[Tuple[int, int]], torch.Tensor]],
    nheads: int,
    head_dim: int,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes,
    deterministic: bool,
    world_size: int,
    cu_seqlens_k: torch.Tensor,
    chunk_idx_0: int,
    chunk_idx_1: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Drop-in replacement for execute_grouped_attention using fused varlen kernel.

    This function serves as an integration wrapper that converts the inputs from
    the original zigzag implementation format to the fused kernel format.

    Args:
        chunk_q_list: List of Q tensors for each group (length 2)
        chunk_cu_seqlens_q_list: List of cu_seqlens for each group (length 2)
        chunk_indices_list: List of indices for reconstruction (length 2)
        kv_buffer: [2, total_tokens, nheads_k, head_dim] in zigzag format
        kv_slices: List of (seq_ranges, cu_seqlens_k_slice) for each group
        nheads: Number of query heads
        head_dim: Head dimension
        softmax_scale: Scaling factor for attention
        dropout_p: Dropout probability (must be 0 for this implementation)
        causal: Whether to use causal masking
        window_size: Window size (not used in this implementation)
        alibi_slopes: ALiBi slopes (not used in this implementation)
        deterministic: Whether to use deterministic mode
        world_size: Number of GPUs
        cu_seqlens_k: [batch_size + 1] - LOCAL cumulative sequence lengths for K
        chunk_idx_0: Global chunk index for Q0
        chunk_idx_1: Global chunk index for Q1

    Returns:
        out: Combined output in original local Q order
        lse: Combined LSE
        chunk_info: Dictionary with chunking information
    """
    from .zigzag_llama3_flash_attn_varlen import rearrange_kv_from_zigzag_to_contiguous

    assert dropout_p == 0, "Fused kernel does not support dropout yet"
    assert len(chunk_q_list) == 2, "Fused kernel only supports n_chunks=2"

    # Extract Q groups
    q0 = chunk_q_list[0]
    q1 = chunk_q_list[1]
    cu_seqlens_q0 = chunk_cu_seqlens_q_list[0]
    cu_seqlens_q1 = chunk_cu_seqlens_q_list[1]

    # Rearrange KV from zigzag to contiguous format
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
    k = kv_contiguous[0]
    v = kv_contiguous[1]

    # Compute max_seqlen
    max_seqlen_q0 = (cu_seqlens_q0[1:] - cu_seqlens_q0[:-1]).max().item()
    max_seqlen_q1 = (cu_seqlens_q1[1:] - cu_seqlens_q1[:-1]).max().item()

    # Build GLOBAL cu_seqlens_k for the contiguous buffer
    cu_seqlens_k_global = cu_seqlens_k * world_size
    max_seqlen_k = (cu_seqlens_k_global[1:] - cu_seqlens_k_global[:-1]).max().item()

    # Compute attention ranges from chunk indices
    total_chunks = 2 * world_size
    chunk_size = max_seqlen_k // total_chunks
    max_kv_len_q0 = (chunk_idx_0 + 1) * chunk_size
    max_kv_len_q1 = (chunk_idx_1 + 1) * chunk_size

    # Call fused kernel
    out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k_global,
        max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
        max_kv_len_q0, max_kv_len_q1,
        softmax_scale, causal,
    )

    # Reconstruct output (scatter back to original positions)
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros((total_q, nheads, head_dim), dtype=out0.dtype, device=out0.device)

    indices0 = chunk_indices_list[0]
    indices1 = chunk_indices_list[1]
    out[indices0] = out0
    out[indices1] = out1

    # Reconstruct LSE
    # lse0 and lse1 are [batch_size * nheads, max_seqlen_rounded]
    # Need to convert to [nheads, total_q]
    batch_size = len(cu_seqlens_q0) - 1
    lse = torch.zeros((nheads, total_q), dtype=lse0.dtype, device=lse0.device)

    # Extract actual LSE values (not padded parts)
    for i in range(batch_size):
        seq_start_q0 = cu_seqlens_q0[i].item()
        seq_end_q0 = cu_seqlens_q0[i+1].item()
        seq_len_q0 = seq_end_q0 - seq_start_q0

        seq_start_q1 = cu_seqlens_q1[i].item()
        seq_end_q1 = cu_seqlens_q1[i+1].item()
        seq_len_q1 = seq_end_q1 - seq_start_q1

        for h in range(nheads):
            hz_idx = i * nheads + h
            # Copy Q0 LSE
            lse_vals_q0 = lse0[hz_idx, :seq_len_q0]
            lse[:, indices0[seq_start_q0:seq_end_q0]] = lse_vals_q0.unsqueeze(0)
            # Copy Q1 LSE
            lse_vals_q1 = lse1[hz_idx, :seq_len_q1]
            lse[:, indices1[seq_start_q1:seq_end_q1]] = lse_vals_q1.unsqueeze(0)

    chunk_info = {
        'chunk_indices_list': chunk_indices_list,
        'chunk_cu_seqlens_q_list': chunk_cu_seqlens_q_list,
        'kv_slices': kv_slices,
    }

    return out, lse, chunk_info
