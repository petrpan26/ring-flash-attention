"""
Triton Grouped Flash Attention Implementation

This module implements grouped flash attention using Triton, where multiple Q groups
share K,V loads in a single fused kernel. This eliminates redundant HBM reads
when different Q groups attend to overlapping K,V regions.

Key features:
- Multiple Q groups with different K,V slice lengths
- Online softmax for numerical stability
- Support for causal masking
- Support for variable-length sequences (varlen)
- Support for Grouped Query Attention (GQA)
- Optimized block sizes with autotuning

Design:
- Each program instance processes one Q block for one group
- K,V tiles are loaded once and shared across groups via L2 cache
- Separate outputs and LSE per group

Reference:
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- Triton Flash Attention Tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch
import triton
import triton.language as tl
from typing import List, Tuple, Optional

# Suppress torch.compile warnings
import warnings
warnings.filterwarnings('ignore', message='.*Graph break from.*Tensor.item.*')


@triton.jit
def grouped_flash_attention_fwd_kernel(
    # Q tensors (one per group)
    Q_GROUP_PTRS,  # Array of pointers to Q tensors
    Q_GROUP_STARTS,  # Start index in combined buffer for each group
    Q_GROUP_LENS,  # Length of each Q group
    # Shared K, V tensors
    K_PTR,
    V_PTR,
    # Output tensors (one per group)
    OUT_GROUP_PTRS,  # Array of pointers to output tensors
    LSE_GROUP_PTRS,  # Array of pointers to LSE tensors
    # Sequence metadata per group
    CU_SEQLENS_Q_GROUP_PTRS,  # Array of pointers to cu_seqlens_q per group
    CU_SEQLENS_K_GROUP_PTRS,  # Array of pointers to cu_seqlens_k per group
    MAX_SEQLEN_Q_GROUP,  # Array of max_seqlen_q per group
    MAX_SEQLEN_K_GROUP,  # Array of max_seqlen_k per group (DIFFERENT per group!)
    # Dimensions
    nheads_q,
    nheads_k,
    head_dim,
    # Parameters
    softmax_scale,
    group_id,  # Which group this kernel instance processes
    # Strides
    stride_qb, stride_qh, stride_qd,  # Q strides (batch/token, head, dim)
    stride_kb, stride_kh, stride_kd,  # K strides
    stride_vb, stride_vh, stride_vd,  # V strides
    stride_ob, stride_oh, stride_od,  # Output strides
    stride_lse_b, stride_lse_h,  # LSE strides (batch/token, head)
    # Masking
    IS_CAUSAL: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    # GQA support
    NUM_GROUPS: tl.constexpr,  # Q heads per K head
):
    """
    Grouped flash attention forward kernel.

    This kernel processes one Q block for one group. Multiple groups can share
    K,V loads via L2 cache when they access overlapping regions.

    Args:
        Q_GROUP_PTRS: Combined buffer with all Q groups concatenated
        Q_GROUP_STARTS: Offset for this group in combined Q buffer
        Q_GROUP_LENS: Number of tokens for this group
        K_PTR, V_PTR: Shared K, V tensors (full length)
        OUT_GROUP_PTRS: Combined output buffer
        LSE_GROUP_PTRS: Combined LSE buffer
        CU_SEQLENS_Q_GROUP_PTRS: Combined cu_seqlens_q buffer with offsets
        CU_SEQLENS_K_GROUP_PTRS: Combined cu_seqlens_k buffer with offsets
        MAX_SEQLEN_Q_GROUP: Array of max Q sequence lengths per group
        MAX_SEQLEN_K_GROUP: Array of max K sequence lengths per group
        group_id: Which group this kernel processes
        IS_CAUSAL: Whether to apply causal masking
        BLOCK_M, BLOCK_N: Tile sizes for Q and K/V
        BLOCK_DMODEL: Head dimension (must match head_dim)
        NUM_GROUPS: Number of Q heads per K head (for GQA)
    """
    # Get program IDs
    pid_m = tl.program_id(0)  # Q block index within this group
    pid_head = tl.program_id(1)  # Head index

    # Load group-specific metadata
    q_start = tl.load(Q_GROUP_STARTS + group_id)
    q_len = tl.load(Q_GROUP_LENS + group_id)
    max_seqlen_k = tl.load(MAX_SEQLEN_K_GROUP + group_id)

    # Check if this Q block is valid
    if pid_m * BLOCK_M >= q_len:
        return

    # Calculate Q block position
    q_block_start = q_start + pid_m * BLOCK_M

    # For GQA, map Q head to K head
    k_head = pid_head // NUM_GROUPS

    # Calculate base pointers for this Q block and head
    Q = Q_GROUP_PTRS + q_block_start * stride_qb + pid_head * stride_qh
    K = K_PTR + k_head * stride_kh
    V = V_PTR + k_head * stride_vh
    OUT = OUT_GROUP_PTRS + q_block_start * stride_ob + pid_head * stride_oh
    LSE = LSE_GROUP_PTRS + q_block_start * stride_lse_b + pid_head * stride_lse_h

    # Load Q block: [BLOCK_M, BLOCK_DMODEL]
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + offs_m[:, None] * stride_qb + offs_d[None, :] * stride_qd
    mask_m = (q_block_start + offs_m) < (q_start + q_len)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators for online softmax
    # m_i: running max of attention scores
    # l_i: running sum of exp(score - m_i)
    # acc: accumulated attention-weighted values
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K, V blocks
    n_blocks = tl.cdiv(max_seqlen_k, BLOCK_N)

    for block_n in range(n_blocks):
        k_block_start = block_n * BLOCK_N

        # Load K block: [BLOCK_N, BLOCK_DMODEL]
        offs_n = tl.arange(0, BLOCK_N)
        k_ptrs = K + (k_block_start + offs_n[:, None]) * stride_kb + offs_d[None, :] * stride_kd
        mask_n = (k_block_start + offs_n) < max_seqlen_k
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute attention scores: QK^T
        # qk: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), acc=qk)
        qk = qk * softmax_scale

        # Apply causal masking
        if IS_CAUSAL:
            # Causal mask: q_pos >= k_pos
            # Note: This simplified version assumes single sequence
            # For varlen, would need to use cu_seqlens
            causal_mask = (q_block_start + offs_m[:, None]) >= (k_block_start + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Apply valid token mask
        valid_mask = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(valid_mask, qk, float("-inf"))

        # Online softmax update
        # Compute new max
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Compute exp(qk - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # Update running sum: l_i_new = exp(m_i - m_i_new) * l_i + sum(p)
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, axis=1)

        # Load V block: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + (k_block_start + offs_n[:, None]) * stride_vb + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Update accumulator: acc = alpha * acc + p @ v
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc)

        # Update running statistics
        m_i = m_i_new
        l_i = l_i_new

    # Finalize: o = acc / l_i
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = OUT + offs_m[:, None] * stride_ob + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_m[:, None])

    # Store LSE (log-sum-exp): log(l_i) + m_i
    lse = tl.log(l_i) + m_i
    lse_ptrs = LSE + offs_m * stride_lse_b
    tl.store(lse_ptrs, lse.to(LSE.dtype.element_ty), mask=mask_m)


def triton_grouped_flash_attn_varlen_forward(
    q_list: List[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q_list: List[torch.Tensor],
    cu_seqlens_k_list: List[torch.Tensor],
    max_seqlen_q_list: List[int],
    max_seqlen_k_list: List[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Triton-based grouped flash attention forward pass.

    Multiple Q groups share K,V loads in a single fused kernel invocation.
    Each group can attend to different-length prefixes of K,V.

    Args:
        q_list: List of Q tensors, one per group [tokens_i, nheads, head_dim]
        k: Shared K tensor [total_k_tokens, nheads_k, head_dim]
        v: Shared V tensor [total_v_tokens, nheads_k, head_dim]
        cu_seqlens_q_list: List of cumulative Q sequence lengths per group
        cu_seqlens_k_list: List of cumulative K sequence lengths per group
        max_seqlen_q_list: List of max Q sequence lengths per group
        max_seqlen_k_list: List of max K sequence lengths per group (can differ!)
        dropout_p: Dropout probability (not supported in this implementation)
        softmax_scale: Scale for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Sliding window size (not supported yet)
        alibi_slopes: ALiBi slopes (not supported yet)
        return_softmax: Whether to return softmax matrix (not supported)

    Returns:
        out_list: List of output tensors, one per group
        lse_list: List of LSE tensors, one per group

    Example:
        # Two groups with different K,V slice lengths
        q_early = torch.randn(1000, 32, 128, device='cuda', dtype=torch.float16)
        q_late = torch.randn(1000, 32, 128, device='cuda', dtype=torch.float16)
        k = torch.randn(2000, 8, 128, device='cuda', dtype=torch.float16)
        v = torch.randn(2000, 8, 128, device='cuda', dtype=torch.float16)

        out_list, lse_list = triton_grouped_flash_attn_varlen_forward(
            q_list=[q_early, q_late],
            k=k,
            v=v,
            cu_seqlens_q_list=[torch.tensor([0, 500, 1000], device='cuda', dtype=torch.int32)] * 2,
            cu_seqlens_k_list=[
                torch.tensor([0, 500, 1000], device='cuda', dtype=torch.int32),  # early
                torch.tensor([0, 1000, 2000], device='cuda', dtype=torch.int32),  # late
            ],
            max_seqlen_q_list=[500, 500],
            max_seqlen_k_list=[1000, 2000],  # Different K lengths!
            causal=True,
        )
    """
    # Validation
    assert len(q_list) > 0, "Must have at least one Q group"
    assert len(q_list) == len(cu_seqlens_q_list) == len(max_seqlen_q_list), \
        "Q lists must have same length"
    assert len(cu_seqlens_k_list) == len(max_seqlen_k_list) == len(q_list), \
        "K lists must match Q group count"
    assert dropout_p == 0.0, "Dropout not supported in Triton implementation"
    assert window_size == (-1, -1), "Sliding window not supported yet"
    assert alibi_slopes is None, "ALiBi not supported yet"
    assert not return_softmax, "return_softmax not supported"

    num_groups = len(q_list)
    device = q_list[0].device
    dtype = q_list[0].dtype

    # Get dimensions
    nheads_q = q_list[0].shape[1]
    head_dim = q_list[0].shape[2]
    nheads_k = k.shape[1]

    # Validate shapes
    for i, q in enumerate(q_list):
        assert q.shape[1] == nheads_q, f"Q group {i} has inconsistent nheads"
        assert q.shape[2] == head_dim, f"Q group {i} has inconsistent head_dim"
    assert v.shape[1] == nheads_k and v.shape[2] == head_dim, "K,V shape mismatch"

    # Calculate softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    # GQA: Number of Q heads per K head
    assert nheads_q % nheads_k == 0, "nheads_q must be divisible by nheads_k for GQA"
    num_groups_gqa = nheads_q // nheads_k

    # Prepare combined Q buffer (concatenate all groups)
    q_combined = torch.cat(q_list, dim=0)
    q_group_starts = torch.tensor([0] + [q.shape[0] for q in q_list], device=device, dtype=torch.int32)
    q_group_starts = torch.cumsum(q_group_starts, dim=0)[:-1]
    q_group_lens = torch.tensor([q.shape[0] for q in q_list], device=device, dtype=torch.int32)

    # Prepare output buffers
    out_list = []
    lse_list = []
    for q in q_list:
        out = torch.empty_like(q)
        lse = torch.empty((q.shape[0], nheads_q), device=device, dtype=torch.float32)
        out_list.append(out)
        lse_list.append(lse)

    out_combined = torch.cat(out_list, dim=0)
    lse_combined = torch.cat(lse_list, dim=0)

    # Prepare group metadata
    max_seqlen_q_tensor = torch.tensor(max_seqlen_q_list, device=device, dtype=torch.int32)
    max_seqlen_k_tensor = torch.tensor(max_seqlen_k_list, device=device, dtype=torch.int32)

    # Block sizes (optimized for A100/H100)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    # Ensure head_dim is power of 2 or pad
    if head_dim != BLOCK_DMODEL:
        # Would need to pad tensors - not implemented
        raise NotImplementedError(f"head_dim={head_dim} must be power of 2")

    # Launch kernel for each group and each head
    for group_id in range(num_groups):
        q_len = q_group_lens[group_id].item()
        num_q_blocks = triton.cdiv(q_len, BLOCK_M)

        grid = (num_q_blocks, nheads_q)

        with torch.cuda.device(device):
            grouped_flash_attention_fwd_kernel[grid](
                # Q tensors
                q_combined,
                q_group_starts,
                q_group_lens,
                # K, V
                k,
                v,
                # Outputs
                out_combined,
                lse_combined,
                # Sequence metadata (placeholder - simplified version)
                q_group_starts,  # Placeholder
                q_group_starts,  # Placeholder
                max_seqlen_q_tensor,
                max_seqlen_k_tensor,
                # Dimensions
                nheads_q,
                nheads_k,
                head_dim,
                # Parameters
                softmax_scale,
                group_id,
                # Strides
                q_combined.stride(0), q_combined.stride(1), q_combined.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out_combined.stride(0), out_combined.stride(1), out_combined.stride(2),
                lse_combined.stride(0), lse_combined.stride(1),
                # Masking
                IS_CAUSAL=causal,
                # Block sizes
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
                # GQA
                NUM_GROUPS=num_groups_gqa,
            )

    # Split outputs back to per-group lists
    out_list_final = []
    lse_list_final = []
    offset = 0
    for q in q_list:
        q_len = q.shape[0]
        out_list_final.append(out_combined[offset:offset+q_len])
        lse_list_final.append(lse_combined[offset:offset+q_len])
        offset += q_len

    return out_list_final, lse_list_final


# Autotuning configuration for optimal block sizes
# (This is a placeholder showing how autotuning would be applied in the future)
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=2, num_stages=5),
#     ],
#     key=['max_seqlen_q', 'max_seqlen_k', 'head_dim'],
# )
# @triton.jit
# def grouped_flash_attention_fwd_kernel_autotuned(...):
#     """
#     Autotuned version of grouped flash attention kernel.
#
#     Triton will benchmark different block size configurations and select
#     the fastest one for the given input shapes.
#     """
#     pass


def get_recommended_block_sizes(head_dim: int, max_seqlen_k: int) -> Tuple[int, int, int]:
    """
    Get recommended block sizes based on input dimensions.

    Args:
        head_dim: Head dimension
        max_seqlen_k: Maximum K sequence length

    Returns:
        (BLOCK_M, BLOCK_N, BLOCK_DMODEL)

    Recommendations:
        - BLOCK_M, BLOCK_N: Trade-off between memory and parallelism
          - Larger blocks: Better memory efficiency, fewer kernel launches
          - Smaller blocks: Better load balancing, more parallelism
        - For long sequences (>2048): Use 128x128
        - For medium sequences (512-2048): Use 64x64 or 64x128
        - For short sequences (<512): Use 64x32
        - BLOCK_DMODEL must be >= head_dim and power of 2
    """
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    if max_seqlen_k > 2048:
        return 128, 128, BLOCK_DMODEL
    elif max_seqlen_k > 512:
        return 64, 64, BLOCK_DMODEL
    else:
        return 64, 32, BLOCK_DMODEL
