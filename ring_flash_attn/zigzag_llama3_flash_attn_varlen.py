"""
Zigzag Llama3 Flash Attention - Variable Length Implementation

This implementation combines:
- Zigzag interleaved data distribution (for load balancing in causal attention)
- Llama3-style all-gather communication (simple, one-step)
- Chunk-based Q splitting (split by global chunk index, not position)

See ZIGZAG_LLAMA3_IMPLEMENTATION.md for detailed design documentation.

Key workflow:
1. Input: Q, K, V in zigzag interleaved format
2. All-gather K, V from all ranks
3. Rearrange K, V from interleaved to contiguous format
4. Split local Q by global chunk index (early vs late chunks)
5. Execute attention with appropriate K,V slices
6. Combine outputs back to interleaved format
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from .utils import get_default_args, AllGatherComm as Comm
from .triton_utils import extract_zigzag_kv_slices_for_group


# ============================================================================
# Helper Functions
# ============================================================================

def rearrange_kv_from_zigzag_to_contiguous(
    kv_buffer: torch.Tensor,
    world_size: int,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange all-gathered K,V from zigzag interleaved format to contiguous format.

    After all-gather, K,V are in interleaved zigzag format across ranks:
    [r0_chunk0, r0_chunk1, r1_chunk0, r1_chunk1, ...]
    where each rank contributed [chunk_i, chunk_(2*world_size-1-i)]

    This function converts to contiguous order:
    [chunk0, chunk1, chunk2, ..., chunk(2*world_size-1)]

    Args:
        kv_buffer: [2, total_tokens, heads_k_stride, head_dim]
                   First dim is K/V, tokens are in interleaved order
        world_size: Number of GPUs
        cu_seqlens: [num_seqs + 1] - Cumulative sequence lengths for GLOBAL sequences

    Returns:
        Rearranged kv_buffer in contiguous order [2, total_tokens, heads_k_stride, head_dim]

    Example (world_size=4, local_tokens=256 each):
        Input (interleaved):
        [r0_c0, r0_c7, r1_c1, r1_c6, r2_c2, r2_c5, r3_c3, r3_c4]
        = [c0(128), c7(128), c1(128), c6(128), c2(128), c5(128), c3(128), c4(128)]

        Output (contiguous):
        [c0(128), c1(128), c2(128), c3(128), c4(128), c5(128), c6(128), c7(128)]
    """
    # kv_buffer shape: [2, total_tokens, heads_k_stride, head_dim]
    total_tokens = kv_buffer.shape[1]
    local_tokens = total_tokens // world_size
    num_sequences = len(cu_seqlens) - 1
    total_chunks = 2 * world_size

    # Split into per-rank chunks
    # Each rank contributed local_tokens
    kv_per_rank = kv_buffer.reshape(
        2, world_size, local_tokens, kv_buffer.shape[2], kv_buffer.shape[3]
    )  # [2, world_size, local_tokens, heads_k_stride, head_dim]

    # Process each sequence separately (they may have different lengths)
    contiguous_chunks = []
    offset_per_rank = [0] * world_size  # Track position within each rank's data

    for seq_idx in range(num_sequences):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        seq_len = seq_end - seq_start
        chunk_size = seq_len // total_chunks

        # Extract all chunks for this sequence in order
        for chunk_idx in range(total_chunks):
            # Determine which rank has this chunk
            if chunk_idx < world_size:
                # Early chunks: rank = chunk_idx
                rank = chunk_idx
                is_first_chunk = True
            else:
                # Late chunks: rank = 2*world_size - 1 - chunk_idx
                rank = 2 * world_size - 1 - chunk_idx
                is_first_chunk = False

            # Extract chunk from this rank's data
            start_in_rank = offset_per_rank[rank]
            end_in_rank = start_in_rank + chunk_size
            chunk = kv_per_rank[:, rank, start_in_rank:end_in_rank]
            contiguous_chunks.append(chunk)

            # Update offset only after extracting the second chunk from this rank
            if not is_first_chunk or chunk_idx == rank:
                offset_per_rank[rank] = end_in_rank

    # Concatenate all chunks
    kv_contiguous = torch.cat(contiguous_chunks, dim=1)

    return kv_contiguous


# Optimization #4: torch.compile for hot path functions
@torch.compile(mode="reduce-overhead", fullgraph=False)
def split_q_by_zigzag_chunk_index(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    world_size: int,
    rank: int,
    n_chunks: int = 2
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Split local zigzag-interleaved Q into groups based on global chunk indices.

    The local Q is in zigzag interleaved format. For each sequence, each rank has:
    - chunk[rank] + chunk[2*world_size - 1 - rank]

    For n_chunks=2, we split into:
    - Group 0 (early): chunks [0, 1, ..., world_size-1]
    - Group 1 (late): chunks [world_size, ..., 2*world_size-1]

    Args:
        q: [local_tokens, nheads, head_dim] - Local Q in zigzag interleaved format
        cu_seqlens_q: [num_seqs + 1] - Cumulative sequence lengths for LOCAL q
        world_size: Number of GPUs
        rank: Current rank
        n_chunks: Number of groups (default: 2)

    Returns:
        chunk_q_list: List of Q tensors for each group
        chunk_cu_seqlens_q_list: List of cu_seqlens for each group
        chunk_indices_list: List of local indices for reconstruction

    Example (world_size=4, rank=0, n_chunks=2, 2 sequences):
        cu_seqlens_q = [0, 256, 512]  (each sequence contributes 256 tokens locally)

        Local Q structure (interleaved):
        [seq0_c0, seq0_c7, seq1_c0, seq1_c7]
        [0:128,   128:256, 256:384, 384:512]

        Group 0 (early, chunks 0-3):
            [seq0_c0, seq1_c0]
            cu_seqlens: [0, 128, 256]
            indices: [0:128, 256:384]

        Group 1 (late, chunks 4-7):
            [seq0_c7, seq1_c7]
            cu_seqlens: [0, 128, 256]
            indices: [128:256, 384:512]
    """
    num_sequences = len(cu_seqlens_q) - 1
    total_chunks = 2 * world_size
    chunks_per_group = total_chunks // n_chunks

    # Identify which global chunks this rank has
    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank

    # Determine which group each chunk belongs to
    def get_group(chunk_idx):
        return min(chunk_idx // chunks_per_group, n_chunks - 1)

    group_0 = get_group(chunk_idx_0)
    group_1 = get_group(chunk_idx_1)

    # Initialize lists for each group
    chunk_q_list = [[] for _ in range(n_chunks)]
    chunk_cu_seqlens_list = [[0] for _ in range(n_chunks)]
    chunk_indices_list = [[] for _ in range(n_chunks)]

    # Process each sequence
    for seq_idx in range(num_sequences):
        seq_start = cu_seqlens_q[seq_idx].item()
        seq_end = cu_seqlens_q[seq_idx + 1].item()
        seq_len = seq_end - seq_start

        # Each sequence in local Q has 2 chunks (interleaved)
        local_chunk_size = seq_len // 2

        # First local chunk (chunk_idx_0)
        chunk_0_start = seq_start
        chunk_0_end = seq_start + local_chunk_size
        chunk_q_list[group_0].append(q[chunk_0_start:chunk_0_end])
        chunk_indices_list[group_0].append(torch.arange(
            chunk_0_start, chunk_0_end, device=q.device
        ))
        chunk_cu_seqlens_list[group_0].append(
            chunk_cu_seqlens_list[group_0][-1] + local_chunk_size
        )

        # Second local chunk (chunk_idx_1)
        chunk_1_start = chunk_0_end
        chunk_1_end = seq_end
        chunk_q_list[group_1].append(q[chunk_1_start:chunk_1_end])
        chunk_indices_list[group_1].append(torch.arange(
            chunk_1_start, chunk_1_end, device=q.device
        ))
        chunk_cu_seqlens_list[group_1].append(
            chunk_cu_seqlens_list[group_1][-1] + local_chunk_size
        )

    # Concatenate tensors in each group
    final_chunk_q_list = []
    final_chunk_cu_seqlens_list = []
    final_chunk_indices_list = []

    for group_idx in range(n_chunks):
        if chunk_q_list[group_idx]:
            final_chunk_q_list.append(torch.cat(chunk_q_list[group_idx], dim=0))
            final_chunk_cu_seqlens_list.append(torch.tensor(
                chunk_cu_seqlens_list[group_idx],
                device=q.device,
                dtype=cu_seqlens_q.dtype
            ))
            final_chunk_indices_list.append(torch.cat(chunk_indices_list[group_idx]))
        else:
            # Empty group (shouldn't happen with n_chunks=2, but handle gracefully)
            final_chunk_q_list.append(torch.empty(
                0, q.shape[1], q.shape[2], device=q.device, dtype=q.dtype
            ))
            final_chunk_cu_seqlens_list.append(torch.tensor(
                [0], device=q.device, dtype=cu_seqlens_q.dtype
            ))
            final_chunk_indices_list.append(torch.empty(0, device=q.device, dtype=torch.long))

    return final_chunk_q_list, final_chunk_cu_seqlens_list, final_chunk_indices_list


# Optimization #4: torch.compile for hot path functions
@torch.compile(mode="reduce-overhead", fullgraph=False)
def compute_kv_slices_for_groups(
    cu_seqlens_k: torch.Tensor,
    chunk_idx_0: int,
    chunk_idx_1: int,
    world_size: int,
    n_chunks: int = 2
) -> List[Tuple[List[Tuple[int, int]], torch.Tensor]]:
    """
    Compute which portion of contiguous K,V each Q group needs for causal attention.

    For causal attention, a chunk at position i needs K,V from positions [0, i].
    This function determines the K,V slice ranges for each sequence in each group.

    Args:
        cu_seqlens_k: [num_seqs + 1] - Cumulative lengths for contiguous GLOBAL K,V
        chunk_idx_0: Global chunk index for first local chunk
        chunk_idx_1: Global chunk index for second local chunk
        world_size: Number of GPUs
        n_chunks: Number of groups (default: 2)

    Returns:
        List of (seq_ranges, cu_seqlens_k_slice) for each group
        seq_ranges: List of (start, end) tuples for each sequence in contiguous K,V buffer
        cu_seqlens_k_slice: Adjusted cu_seqlens for the extracted K,V slice

    Example (world_size=4, rank=0, n_chunks=2):
        chunk_idx_0 = 0, chunk_idx_1 = 7
        total_chunks = 8

        Group 0 (contains chunk_idx_0=0): needs K,V up to end of chunk 0 for each seq
        Group 1 (contains chunk_idx_1=7): needs K,V up to end of chunk 7 (full) for each seq
    """
    num_sequences = len(cu_seqlens_k) - 1
    total_chunks = 2 * world_size
    chunks_per_group = total_chunks // n_chunks

    # Determine which group each local chunk belongs to
    def get_group(chunk_idx):
        return min(chunk_idx // chunks_per_group, n_chunks - 1)

    group_0 = get_group(chunk_idx_0)
    group_1 = get_group(chunk_idx_1)

    # For each group, find which of THIS RANK's chunks belongs to it
    # We have chunk_idx_0 and chunk_idx_1 from this rank
    # Each belongs to a group, and we need K,V based on that specific chunk's position
    rank_chunks_in_group = {}
    if group_0 < n_chunks:
        rank_chunks_in_group[group_0] = chunk_idx_0
    if group_1 < n_chunks and group_1 != group_0:
        rank_chunks_in_group[group_1] = chunk_idx_1
    elif group_1 < n_chunks and group_1 == group_0:
        # Both chunks in same group, use the later one (higher position)
        rank_chunks_in_group[group_0] = max(chunk_idx_0, chunk_idx_1)

    # Compute K,V slices for each group
    kv_slices = []
    for group_idx in range(n_chunks):
        # Use THIS RANK's chunk index in this group, not the group maximum
        # This ensures K,V length matches Q length for suffix alignment
        max_chunk_idx = rank_chunks_in_group.get(group_idx, 0)

        # Calculate K,V slice ranges for each sequence
        # For causal attention, need K,V from start up to and including max_chunk_idx
        seq_ranges = []
        cu_seqlens_k_slice = [0]
        cumulative_offset = 0

        for seq_idx in range(num_sequences):
            seq_start = cu_seqlens_k[seq_idx].item()
            seq_end = cu_seqlens_k[seq_idx + 1].item()
            seq_len = seq_end - seq_start

            # Calculate chunk size for this sequence
            chunk_size = seq_len // total_chunks

            # Need tokens from start of sequence up to end of chunk max_chunk_idx
            # In contiguous buffer, this sequence starts at seq_start
            num_tokens_needed = (max_chunk_idx + 1) * chunk_size
            slice_start = seq_start
            slice_end = seq_start + num_tokens_needed

            seq_ranges.append((slice_start, slice_end))

            # Update cumulative cu_seqlens for the extracted slice
            cumulative_offset += num_tokens_needed
            cu_seqlens_k_slice.append(cumulative_offset)

        cu_seqlens_k_slice_tensor = torch.tensor(
            cu_seqlens_k_slice,
            device=cu_seqlens_k.device,
            dtype=cu_seqlens_k.dtype
        )

        kv_slices.append((seq_ranges, cu_seqlens_k_slice_tensor))

    return kv_slices


# ============================================================================
# Forward Execution Modes
# ============================================================================

def execute_two_kernels_mode(
    chunk_q_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    chunk_indices_list: List[torch.Tensor],
    kv_buffer: torch.Tensor,  # Changed: now takes zigzag buffer directly
    kv_slices: List[Tuple[int, torch.Tensor]],
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
    use_triton_kernel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Execute attention with two sequential kernel calls (one per group).

    Simpler approach: Run attention separately for each Q group.

    Args:
        kv_buffer: [2, total_tokens, nheads_k, head_dim] - zigzag interleaved buffer
        use_triton_kernel: If True, use Triton kernel to extract slices directly.
                           If False, fall back to Python rearrangement (for debugging).

    Returns:
        out: Combined output in original local Q order
        lse: Combined LSE
        chunk_info: Dictionary with chunking information for backward
    """
    out_chunks = []
    lse_chunks = []

    for group_idx, (q_group, cu_seqlens_q_group) in enumerate(zip(chunk_q_list, chunk_cu_seqlens_q_list)):
        if q_group.shape[0] == 0:
            # Empty group, skip
            continue

        seq_ranges, cu_seqlens_k_slice = kv_slices[group_idx]

        if use_triton_kernel:
            # Use Triton kernel to extract K,V slices directly from zigzag buffer
            # Build seq_ranges_with_chunk_idx: [(start, end, max_chunk_idx), ...]
            seq_ranges_with_chunk_idx = []
            for seq_idx, (start, end) in enumerate(seq_ranges):
                # Calculate max_chunk_idx for this sequence
                seq_start_global = cu_seqlens_k[seq_idx].item()
                seq_end_global = cu_seqlens_k[seq_idx + 1].item()
                seq_len = seq_end_global - seq_start_global
                total_chunks = 2 * world_size
                chunk_size = seq_len // total_chunks

                # Number of tokens needed = end - start
                # max_chunk_idx = (num_tokens_needed // chunk_size) - 1
                num_tokens_needed = end - start
                max_chunk_idx = (num_tokens_needed // chunk_size) - 1
                seq_ranges_with_chunk_idx.append((start, end, max_chunk_idx))

            nheads_k = kv_buffer.shape[2]
            k_slice, v_slice = extract_zigzag_kv_slices_for_group(
                kv_buffer,
                seq_ranges_with_chunk_idx,
                cu_seqlens_k,
                world_size,
                nheads_k,
                head_dim,
            )
        else:
            # Fallback: Python-based extraction (original code)
            # First rearrange to contiguous format
            kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)

            # Extract K,V slices for each sequence and concatenate
            k_slices = []
            v_slices = []
            for start, end in seq_ranges:
                k_slices.append(kv_contiguous[0, start:end])
                v_slices.append(kv_contiguous[1, start:end])

            k_slice = torch.cat(k_slices, dim=0)  # [total_k_tokens, heads, dim]
            v_slice = torch.cat(v_slices, dim=0)  # [total_v_tokens, heads, dim]

        # Calculate max_seqlen
        max_seqlen_q = (cu_seqlens_q_group[1:] - cu_seqlens_q_group[:-1]).max().item()
        max_seqlen_k = (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()

        # Flash attention forward
        params = get_default_args(_flash_attn_varlen_forward).copy()
        params.update({
            "q": q_group,
            "k": k_slice,
            "v": v_slice,
            "cu_seqlens_q": cu_seqlens_q_group,
            "cu_seqlens_k": cu_seqlens_k_slice,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        })

        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update({
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            })

        outputs = _flash_attn_varlen_forward(**params)
        if len(outputs) == 8:
            out_group, _, _, _, _, lse_group, _, _ = outputs
        else:
            assert len(outputs) == 4
            out_group, lse_group, _, _ = outputs

        out_chunks.append(out_group)
        lse_chunks.append(lse_group)

    # Combine outputs back to original local Q order
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros(
        (total_q, nheads, head_dim),
        dtype=out_chunks[0].dtype,
        device=out_chunks[0].device
    )

    # Scatter each chunk back to its original positions
    for out_chunk, indices in zip(out_chunks, chunk_indices_list):
        out[indices] = out_chunk

    # Scatter LSE similarly
    if lse_chunks[0].dim() == 2:
        # LSE is [nheads, chunk_len]
        lse = torch.zeros(
            (nheads, total_q),
            dtype=lse_chunks[0].dtype,
            device=lse_chunks[0].device
        )
        for lse_chunk, indices in zip(lse_chunks, chunk_indices_list):
            lse[:, indices] = lse_chunk
    else:
        # LSE is [chunk_len, nheads]
        lse = torch.zeros(
            (total_q, nheads),
            dtype=lse_chunks[0].dtype,
            device=lse_chunks[0].device
        )
        for lse_chunk, indices in zip(lse_chunks, chunk_indices_list):
            lse[indices] = lse_chunk

    chunk_info = {
        'chunk_indices_list': chunk_indices_list,
        'chunk_cu_seqlens_q_list': chunk_cu_seqlens_q_list,
        'kv_slices': kv_slices,
    }

    return out, lse, chunk_info


def execute_fused_kernel_mode(
    chunk_q_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    chunk_indices_list: List[torch.Tensor],
    kv_contiguous: torch.Tensor,
    kv_slices: List[Tuple[int, torch.Tensor]],
    nheads: int,
    head_dim: int,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Execute attention with one fused kernel call (all groups together).

    More efficient (~5% faster): Concatenate all Q groups and duplicate K,V slices.

    Returns:
        out: Combined output in original local Q order
        lse: Combined LSE
        chunk_info: Dictionary with chunking information for backward
    """
    # Concatenate all Q groups
    q_combined = torch.cat(chunk_q_list, dim=0)

    # Create duplicated K,V buffer
    k_dup_list = []
    v_dup_list = []
    cu_seqlens_k_list = [0]

    for seq_ranges, cu_seqlens_k_slice in kv_slices:
        # Extract K,V slices for each sequence and concatenate
        k_slices = []
        v_slices = []
        for start, end in seq_ranges:
            k_slices.append(kv_contiguous[0, start:end])
            v_slices.append(kv_contiguous[1, start:end])

        k_slice = torch.cat(k_slices, dim=0)
        v_slice = torch.cat(v_slices, dim=0)
        k_dup_list.append(k_slice)
        v_dup_list.append(v_slice)
        cu_seqlens_k_list.extend((cu_seqlens_k_slice[1:] + cu_seqlens_k_list[-1]).tolist())

    k_duplicated = torch.cat(k_dup_list, dim=0)
    v_duplicated = torch.cat(v_dup_list, dim=0)
    cu_seqlens_k_combined = torch.tensor(
        cu_seqlens_k_list[:len(chunk_q_list) * (len(kv_slices[0][1]) - 1) + 1],
        device=kv_contiguous.device,
        dtype=torch.int32
    )

    # Combine cu_seqlens_q
    cu_seqlens_q_combined = [0]
    for chunk_cu_seqlens in chunk_cu_seqlens_q_list:
        cu_seqlens_q_combined.extend((chunk_cu_seqlens[1:] + cu_seqlens_q_combined[-1]).tolist())

    cu_seqlens_q_combined = torch.tensor(
        cu_seqlens_q_combined,
        device=q_combined.device,
        dtype=torch.int32
    )

    # Calculate max_seqlen
    max_seqlen_q_combined = max(
        (chunk_cu[1:] - chunk_cu[:-1]).max().item()
        for chunk_cu in chunk_cu_seqlens_q_list
    )
    max_seqlen_k_combined = max(
        (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()
        for _, cu_seqlens_k_slice in kv_slices
    )

    # Single flash attention forward call
    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update({
        "q": q_combined,
        "k": k_duplicated,
        "v": v_duplicated,
        "cu_seqlens_q": cu_seqlens_q_combined,
        "cu_seqlens_k": cu_seqlens_k_combined,
        "max_seqlen_q": max_seqlen_q_combined,
        "max_seqlen_k": max_seqlen_k_combined,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "alibi_slopes": alibi_slopes,
        "return_softmax": True and dropout_p > 0,
    })

    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update({
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
        })

    outputs = _flash_attn_varlen_forward(**params)
    if len(outputs) == 8:
        out_combined, _, _, _, _, lse_combined, _, _ = outputs
    else:
        assert len(outputs) == 4
        out_combined, lse_combined, _, _ = outputs

    # Scatter output back to original Q order
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros(
        (total_q, nheads, head_dim),
        dtype=out_combined.dtype,
        device=out_combined.device
    )

    offset = 0
    for indices in chunk_indices_list:
        chunk_size = len(indices)
        out[indices] = out_combined[offset:offset+chunk_size]
        offset += chunk_size

    # Scatter LSE similarly
    if lse_combined.dim() == 2 and lse_combined.shape[0] == nheads:
        # LSE is [nheads, total_len]
        lse = torch.zeros(
            (nheads, total_q),
            dtype=lse_combined.dtype,
            device=lse_combined.device
        )
        offset = 0
        for indices in chunk_indices_list:
            chunk_size = len(indices)
            lse[:, indices] = lse_combined[:, offset:offset+chunk_size]
            offset += chunk_size
    else:
        # LSE is [total_len, nheads]
        lse = torch.zeros(
            (total_q, nheads),
            dtype=lse_combined.dtype,
            device=lse_combined.device
        )
        offset = 0
        for indices in chunk_indices_list:
            chunk_size = len(indices)
            lse[indices] = lse_combined[offset:offset+chunk_size]
            offset += chunk_size

    chunk_info = {
        'chunk_indices_list': chunk_indices_list,
        'chunk_cu_seqlens_q_list': chunk_cu_seqlens_q_list,
        'kv_slices': kv_slices,
    }

    return out, lse, chunk_info


# ============================================================================
# Main Forward Function
# ============================================================================

def zigzag_llama3_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    heads_k_stride: int,
    local_k_slice: slice,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes=None,
    deterministic: bool = False,
    use_fused_kernel: bool = False,
    n_chunks: int = 2,
    use_triton_kernel: bool = True,
):
    """
    Forward pass with zigzag-style Q chunking and llama3-style all-gather.

    Process:
    1. Receive Q, K, V in zigzag interleaved format
    2. All-gather K,V from all ranks (llama3-style)
    3. [OPTIMIZED] Use Triton kernel to extract K,V slices directly from zigzag buffer
    4. Split local Q by global chunk index (early vs late groups)
    5. Compute K,V slices for each group (causal attention)
    6. Execute attention (two-kernels or fused mode)
    7. Return output in interleaved format

    Args:
        q, k, v: Input tensors in zigzag interleaved format
        cu_seqlens_q: Cumulative sequence lengths for LOCAL Q
        cu_seqlens_k: Cumulative sequence lengths for GLOBAL K (for slicing)
        use_fused_kernel: If False, use two sequential kernels.
                          If True, use one kernel with duplicated K,V.
        n_chunks: Number of Q groups (default: 2 = early/late)
        use_triton_kernel: If True, use Triton kernel for KV slicing (faster).
                           If False, use Python rearrangement (for debugging).

    Returns:
        out: Output tensor in zigzag interleaved format
        lse: Log-sum-exp values
        chunk_info: Dictionary with chunking information for backward
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape

    # Step 1: Start all-gather K,V (llama3-style) - NON-BLOCKING
    # Optimization #2: Pipeline all-gather with computation
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    k_slice = k[:, :heads_k_stride].contiguous()
    v_slice = v[:, :heads_k_stride].contiguous()

    comm = Comm(process_group)
    comm.all_gather(kv_buffer[0], k_slice)
    comm.all_gather(kv_buffer[1], v_slice)
    # DON'T wait yet - overlap with computation!

    # Step 2: Split local Q by global chunk index (OVERLAPPED with all-gather)
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q, cu_seqlens_q, world_size, rank, n_chunks=n_chunks
    )

    # Step 3: Compute K,V slices for each group (OVERLAPPED with all-gather)
    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    kv_slices = compute_kv_slices_for_groups(
        cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size, n_chunks=n_chunks
    )

    # Now wait for all-gather to complete
    comm.wait()

    # Step 4: Execute attention
    if use_fused_kernel:
        # Fused kernel mode: still needs contiguous rearrangement for now
        kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
        out, lse, chunk_info = execute_fused_kernel_mode(
            chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
            kv_contiguous, kv_slices,
            nheads, head_dim, softmax_scale, dropout_p, causal,
            window_size, alibi_slopes, deterministic
        )
    else:
        # Two-kernels mode: use Triton kernel for direct slicing
        out, lse, chunk_info = execute_two_kernels_mode(
            chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
            kv_buffer, kv_slices,
            nheads, head_dim, softmax_scale, dropout_p, causal,
            window_size, alibi_slopes, deterministic,
            world_size, cu_seqlens_k, use_triton_kernel
        )

    return out, lse, chunk_info


# ============================================================================
# Backward Pass Helper Functions
# ============================================================================

def rearrange_grad_from_contiguous_to_zigzag(
    grad_contiguous: torch.Tensor,
    world_size: int,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange gradients from contiguous to zigzag interleaved format.

    This is the reverse operation of rearrange_kv_from_zigzag_to_contiguous.

    Args:
        grad_contiguous: [2, total_tokens, heads, dim] in contiguous order
        world_size: Number of GPUs
        cu_seqlens: [num_seqs + 1] - Cumulative sequence lengths for GLOBAL sequences

    Returns:
        Gradients in zigzag interleaved format [2, total_tokens, heads, dim]

    Example (world_size=4):
        Input (contiguous): [c0, c1, c2, c3, c4, c5, c6, c7]
        Output (interleaved): [r0_c0, r0_c7, r1_c1, r1_c6, r2_c2, r2_c5, r3_c3, r3_c4]
                            = [c0, c7, c1, c6, c2, c5, c3, c4]
    """
    num_sequences = len(cu_seqlens) - 1
    total_chunks = 2 * world_size

    # Process each sequence separately (they may have different lengths)
    grad_interleaved_chunks = [[] for _ in range(world_size)]
    global_pos = 0

    for seq_idx in range(num_sequences):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        seq_len = seq_end - seq_start
        chunk_size = seq_len // total_chunks

        # For each chunk in this sequence, determine which rank it belongs to
        for chunk_idx in range(total_chunks):
            # Determine which rank has this chunk
            if chunk_idx < world_size:
                # Early chunks: rank = chunk_idx
                rank = chunk_idx
            else:
                # Late chunks: rank = 2*world_size - 1 - chunk_idx
                rank = 2 * world_size - 1 - chunk_idx

            # Extract chunk from contiguous gradient
            chunk_start = global_pos
            chunk_end = global_pos + chunk_size
            chunk = grad_contiguous[:, chunk_start:chunk_end]
            grad_interleaved_chunks[rank].append(chunk)

            global_pos += chunk_size

    # Concatenate chunks for each rank
    grad_interleaved_list = []
    for rank_chunks in grad_interleaved_chunks:
        grad_interleaved_list.append(torch.cat(rank_chunks, dim=1))

    return torch.cat(grad_interleaved_list, dim=1)


# ============================================================================
# Backward Execution Modes
# ============================================================================

def backward_two_kernels_mode(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    kv_buffer: torch.Tensor,  # Changed: now takes zigzag buffer directly
    chunk_indices_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    kv_slices: List[Tuple[int, torch.Tensor]],
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes,
    deterministic: bool,
    world_size: int,
    rank: int,
    process_group,
    cu_seqlens_k: torch.Tensor,
    use_triton_kernel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute backward with two sequential kernel calls (one per group).

    Key: ACCUMULATE gradients for overlapping K,V regions.

    Args:
        kv_buffer: [2, total_tokens, nheads_k, head_dim] - zigzag interleaved buffer
        use_triton_kernel: If True, use Triton kernel for KV slicing (faster).

    Returns:
        dq: Gradients for Q in zigzag interleaved format
        dk: Gradients for K in zigzag interleaved format
        dv: Gradients for V in zigzag interleaved format
    """
    # Split dout by chunk index (same pattern as Q splitting)
    dout_groups = []
    out_groups = []
    lse_groups = []

    for indices in chunk_indices_list:
        dout_groups.append(dout[indices].contiguous())
        out_groups.append(out[indices].contiguous())

        # Handle LSE dimension variations
        if lse.dim() == 2:
            lse_groups.append(lse[:, indices].contiguous())
        else:
            lse_groups.append(lse[indices].contiguous())

    # Split Q by chunk index
    q_groups = []
    for indices in chunk_indices_list:
        q_groups.append(q[indices].contiguous())

    # Initialize gradient buffers
    # For contiguous format gradient accumulation
    nheads_k = kv_buffer.shape[2]
    head_dim = kv_buffer.shape[3]
    total_tokens_k = cu_seqlens_k[-1].item()

    dK_buffer = torch.zeros(
        (total_tokens_k, nheads_k, head_dim),
        dtype=kv_buffer.dtype,
        device=kv_buffer.device
    )
    dV_buffer = torch.zeros_like(dK_buffer)
    dq_groups = []

    # Process each group
    for group_idx, (dout_group, q_group, out_group, lse_group, cu_seqlens_q_group) in enumerate(zip(
        dout_groups, q_groups, out_groups, lse_groups, chunk_cu_seqlens_q_list
    )):
        if q_group.shape[0] == 0:
            # Empty group, skip
            continue

        seq_ranges, cu_seqlens_k_slice = kv_slices[group_idx]

        if use_triton_kernel:
            # Use Triton kernel to extract K,V slices directly from zigzag buffer
            seq_ranges_with_chunk_idx = []
            for seq_idx, (start, end) in enumerate(seq_ranges):
                seq_start_global = cu_seqlens_k[seq_idx].item()
                seq_end_global = cu_seqlens_k[seq_idx + 1].item()
                seq_len = seq_end_global - seq_start_global
                total_chunks = 2 * world_size
                chunk_size = seq_len // total_chunks

                num_tokens_needed = end - start
                max_chunk_idx = (num_tokens_needed // chunk_size) - 1
                seq_ranges_with_chunk_idx.append((start, end, max_chunk_idx))

            k_slice, v_slice = extract_zigzag_kv_slices_for_group(
                kv_buffer,
                seq_ranges_with_chunk_idx,
                cu_seqlens_k,
                world_size,
                nheads_k,
                head_dim,
            )
        else:
            # Fallback: Python-based extraction
            kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
            k_slices = []
            v_slices = []
            for start, end in seq_ranges:
                k_slices.append(kv_contiguous[0, start:end])
                v_slices.append(kv_contiguous[1, start:end])

            k_slice = torch.cat(k_slices, dim=0).contiguous()
            v_slice = torch.cat(v_slices, dim=0).contiguous()

        # Calculate max_seqlen
        max_seqlen_q = (cu_seqlens_q_group[1:] - cu_seqlens_q_group[:-1]).max().item()
        max_seqlen_k = (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()

        # Flash attention backward
        params = get_default_args(_flash_attn_varlen_backward).copy()
        params.update({
            "dout": dout_group,
            "q": q_group,
            "k": k_slice,
            "v": v_slice,
            "out": out_group,
            "softmax_lse": lse_group,
            "dq": torch.empty_like(q_group),
            "dk": torch.empty_like(k_slice),
            "dv": torch.empty_like(v_slice),
            "cu_seqlens_q": cu_seqlens_q_group,
            "cu_seqlens_k": cu_seqlens_k_slice,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
        })

        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update({
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            })

        _flash_attn_varlen_backward(**params)

        dq_group = params["dq"]
        dk_slice = params["dk"]
        dv_slice = params["dv"]

        # CRITICAL: ACCUMULATE gradients (not overwrite!)
        # Multiple Q groups may attend to overlapping K,V regions
        # Scatter gradients back to correct sequence positions
        offset = 0
        for start, end in seq_ranges:
            length = end - start
            dK_buffer[start:end] += dk_slice[offset:offset+length]
            dV_buffer[start:end] += dv_slice[offset:offset+length]
            offset += length

        dq_groups.append(dq_group)

    # Combine dq back to interleaved format
    dq = torch.zeros_like(q)
    for dq_group, indices in zip(dq_groups, chunk_indices_list):
        dq[indices] = dq_group

    # Convert gradients to zigzag interleaved format
    dkv_contiguous = torch.stack([dK_buffer, dV_buffer])  # [2, total_tokens, heads, dim]
    dkv_interleaved = rearrange_grad_from_contiguous_to_zigzag(dkv_contiguous, world_size, cu_seqlens_k)

    # Reduce-scatter to get local gradients
    # Optimization #3: Use reduce_scatter_tensor (newer, more efficient API)
    local_tokens = dkv_interleaved.shape[1] // world_size
    nheads_k_used = dkv_interleaved.shape[2]

    # Allocate output buffers
    local_dk = torch.empty(
        (local_tokens, nheads_k_used, head_dim),
        dtype=dkv_interleaved.dtype,
        device=dkv_interleaved.device
    )
    local_dv = torch.empty_like(local_dk)

    # Optimization #7: Async reduce-scatter to overlap dK and dV communication
    # Start dK reduce-scatter (non-blocking)
    handle_dk = dist.reduce_scatter_tensor(
        local_dk,
        dkv_interleaved[0],
        op=dist.ReduceOp.SUM,
        group=process_group,
        async_op=True
    )

    # Start dV reduce-scatter (can overlap with dK)
    handle_dv = dist.reduce_scatter_tensor(
        local_dv,
        dkv_interleaved[1],
        op=dist.ReduceOp.SUM,
        group=process_group,
        async_op=True
    )

    # Wait for both to complete
    handle_dk.wait()
    handle_dv.wait()

    # Expand back to full head dimension if needed
    if k.shape[1] > local_dk.shape[1]:
        full_dk = torch.zeros_like(k)
        full_dv = torch.zeros_like(v)
        full_dk[:, :local_dk.shape[1]] = local_dk
        full_dv[:, :local_dv.shape[1]] = local_dv
        local_dk = full_dk
        local_dv = full_dv

    return dq, local_dk, local_dv


def backward_fused_kernel_mode(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    kv_contiguous: torch.Tensor,
    chunk_indices_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    kv_slices: List[Tuple[int, torch.Tensor]],
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes,
    deterministic: bool,
    world_size: int,
    rank: int,
    process_group,
    cu_seqlens_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute backward with one fused kernel call (all groups together).

    More efficient: Single backward call, then extract and accumulate gradients.

    Returns:
        dq: Gradients for Q in zigzag interleaved format
        dk: Gradients for K in zigzag interleaved format
        dv: Gradients for V in zigzag interleaved format
    """
    # Split Q by chunk index
    q_groups = []
    for indices in chunk_indices_list:
        q_groups.append(q[indices].contiguous())

    # Concatenate dout, out, and Q (same order as forward)
    dout_combined = torch.cat([dout[indices] for indices in chunk_indices_list], dim=0)
    out_combined = torch.cat([out[indices] for indices in chunk_indices_list], dim=0)
    q_combined = torch.cat(q_groups, dim=0)

    # Handle LSE concatenation
    if lse.dim() == 2:
        lse_combined = torch.cat([lse[:, indices] for indices in chunk_indices_list], dim=1)
    else:
        lse_combined = torch.cat([lse[indices] for indices in chunk_indices_list], dim=0)

    # Create duplicated K,V (same as forward)
    k_dup_list = []
    v_dup_list = []
    cu_seqlens_k_list = [0]

    for seq_ranges, cu_seqlens_k_slice in kv_slices:
        # Extract K,V slices for each sequence and concatenate
        k_slices = []
        v_slices = []
        for start, end in seq_ranges:
            k_slices.append(kv_contiguous[0, start:end])
            v_slices.append(kv_contiguous[1, start:end])

        k_slice = torch.cat(k_slices, dim=0)
        v_slice = torch.cat(v_slices, dim=0)
        k_dup_list.append(k_slice)
        v_dup_list.append(v_slice)
        cu_seqlens_k_list.extend((cu_seqlens_k_slice[1:] + cu_seqlens_k_list[-1]).tolist())

    k_duplicated = torch.cat(k_dup_list, dim=0)
    v_duplicated = torch.cat(v_dup_list, dim=0)

    # Truncate cu_seqlens_k_list to correct length
    num_sequences = sum(len(cu_seq) - 1 for cu_seq in chunk_cu_seqlens_q_list)
    cu_seqlens_k_combined = torch.tensor(
        cu_seqlens_k_list[:num_sequences + 1],
        device=kv_contiguous.device,
        dtype=torch.int32
    )

    # Combine cu_seqlens_q
    cu_seqlens_q_combined = [0]
    for chunk_cu_seqlens in chunk_cu_seqlens_q_list:
        cu_seqlens_q_combined.extend((chunk_cu_seqlens[1:] + cu_seqlens_q_combined[-1]).tolist())
    cu_seqlens_q_combined = torch.tensor(
        cu_seqlens_q_combined,
        device=q_combined.device,
        dtype=torch.int32
    )

    # Calculate max_seqlen
    max_seqlen_q_combined = max(
        (chunk_cu[1:] - chunk_cu[:-1]).max().item()
        for chunk_cu in chunk_cu_seqlens_q_list
    )
    max_seqlen_k_combined = max(
        (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()
        for _, cu_seqlens_k_slice in kv_slices
    )

    # Single flash attention backward call
    params = get_default_args(_flash_attn_varlen_backward).copy()
    params.update({
        "dout": dout_combined,
        "q": q_combined,
        "k": k_duplicated,
        "v": v_duplicated,
        "out": out_combined,
        "softmax_lse": lse_combined,
        "dq": torch.empty_like(q_combined),
        "dk": torch.empty_like(k_duplicated),
        "dv": torch.empty_like(v_duplicated),
        "cu_seqlens_q": cu_seqlens_q_combined,
        "cu_seqlens_k": cu_seqlens_k_combined,
        "max_seqlen_q": max_seqlen_q_combined,
        "max_seqlen_k": max_seqlen_k_combined,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "alibi_slopes": alibi_slopes,
    })

    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update({
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
        })

    _flash_attn_varlen_backward(**params)

    dq_combined = params["dq"]
    dk_duplicated = params["dk"]
    dv_duplicated = params["dv"]

    # Extract and ACCUMULATE dK, dV from duplicated gradients
    dK_buffer = torch.zeros(
        (kv_contiguous.shape[1], kv_contiguous.shape[2], kv_contiguous.shape[3]),
        dtype=kv_contiguous.dtype,
        device=kv_contiguous.device
    )
    dV_buffer = torch.zeros_like(dK_buffer)

    offset_k = 0
    for seq_ranges, cu_seqlens_k_slice in kv_slices:
        # Calculate total length from cu_seqlens
        k_slice_len = cu_seqlens_k_slice[-1].item()
        dk_slice = dk_duplicated[offset_k:offset_k+k_slice_len]
        dv_slice = dv_duplicated[offset_k:offset_k+k_slice_len]

        # ACCUMULATE (different groups may have overlapping K,V regions)
        # Scatter gradients back to correct sequence positions
        offset_in_slice = 0
        for start, end in seq_ranges:
            length = end - start
            dK_buffer[start:end] += dk_slice[offset_in_slice:offset_in_slice+length]
            dV_buffer[start:end] += dv_slice[offset_in_slice:offset_in_slice+length]
            offset_in_slice += length

        offset_k += k_slice_len

    # Scatter dq back to interleaved format
    dq = torch.zeros_like(q)
    offset = 0
    for indices in chunk_indices_list:
        chunk_size = len(indices)
        dq[indices] = dq_combined[offset:offset+chunk_size]
        offset += chunk_size

    # Convert and reduce-scatter
    dkv_contiguous = torch.stack([dK_buffer, dV_buffer])
    dkv_interleaved = rearrange_grad_from_contiguous_to_zigzag(dkv_contiguous, world_size, cu_seqlens_k)

    # Reduce-scatter
    # Optimization #3: Use reduce_scatter_tensor (newer, more efficient API)
    local_tokens = dkv_interleaved.shape[1] // world_size
    nheads_k_used = kv_contiguous.shape[2]
    head_dim = kv_contiguous.shape[3]

    # Allocate output buffers
    local_dk = torch.empty(
        (local_tokens, nheads_k_used, head_dim),
        dtype=dkv_interleaved.dtype,
        device=dkv_interleaved.device
    )
    local_dv = torch.empty_like(local_dk)

    # Optimization #7: Async reduce-scatter to overlap dK and dV communication
    # Start dK reduce-scatter (non-blocking)
    handle_dk = dist.reduce_scatter_tensor(
        local_dk,
        dkv_interleaved[0],
        op=dist.ReduceOp.SUM,
        group=process_group,
        async_op=True
    )

    # Start dV reduce-scatter (can overlap with dK)
    handle_dv = dist.reduce_scatter_tensor(
        local_dv,
        dkv_interleaved[1],
        op=dist.ReduceOp.SUM,
        group=process_group,
        async_op=True
    )

    # Wait for both to complete
    handle_dk.wait()
    handle_dv.wait()

    # Expand back to full head dimension if needed
    if k.shape[1] > local_dk.shape[1]:
        full_dk = torch.zeros_like(k)
        full_dv = torch.zeros_like(v)
        full_dk[:, :local_dk.shape[1]] = local_dk
        full_dv[:, :local_dv.shape[1]] = local_dv
        local_dk = full_dk
        local_dv = full_dv

    return dq, local_dk, local_dv


# ============================================================================
# Main Backward Function
# ============================================================================

def zigzag_llama3_flash_attn_varlen_backward(
    process_group,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    heads_k_stride: int,
    local_k_slice: slice,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes=None,
    deterministic: bool = False,
    use_fused_kernel: bool = False,
    n_chunks: int = 2,
    chunk_indices_list: List[torch.Tensor] = None,
    chunk_cu_seqlens_q_list: List[torch.Tensor] = None,
    kv_slices: List[Tuple[int, torch.Tensor]] = None,
    use_triton_kernel: bool = True,
):
    """
    Backward pass with zigzag-style Q chunking and llama3-style all-gather.

    Process:
    1. Receive dout in zigzag interleaved format
    2. All-gather K, V (needed for gradient computation)
    3. [OPTIMIZED] Use Triton kernel to extract K,V slices directly from zigzag buffer
    4. Split dout by chunk index
    5. Execute backward (two-kernels or fused mode)
    6. ACCUMULATE dK, dV gradients for overlapping regions
    7. Convert gradients to zigzag interleaved
    8. Reduce-scatter dK, dV

    Args:
        dout: Output gradients in zigzag interleaved format
        q, k, v: Forward input tensors
        out, lse: Forward output tensors
        chunk_indices_list: From forward pass
        chunk_cu_seqlens_q_list: From forward pass
        kv_slices: From forward pass
        use_fused_kernel: If False, use two sequential kernels.
                          If True, use one fused kernel.
        use_triton_kernel: If True, use Triton kernel for KV slicing (faster).

    Returns:
        dq: Gradients for Q in zigzag interleaved format
        dk: Gradients for K in zigzag interleaved format
        dv: Gradients for V in zigzag interleaved format
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    total_k, nheads_k, head_dim = k.shape

    # Step 1: Start all-gather K,V (same as forward) - NON-BLOCKING
    # Optimization #2: Pipeline all-gather with computation
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    k_slice = k[:, :heads_k_stride].contiguous()
    v_slice = v[:, :heads_k_stride].contiguous()

    comm = Comm(process_group)
    comm.all_gather(kv_buffer[0], k_slice)
    comm.all_gather(kv_buffer[1], v_slice)
    # DON'T wait yet - can do prep work while waiting

    # TODO: Could add more overlap here (e.g., prepare dout splits)
    # For now, wait before backward computation
    comm.wait()

    # Step 2: Execute backward
    if use_fused_kernel:
        # Fused kernel mode: still needs contiguous rearrangement
        kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
        dq, dk, dv = backward_fused_kernel_mode(
            dout, q, k, v, out, lse,
            kv_contiguous, chunk_indices_list, chunk_cu_seqlens_q_list, kv_slices,
            softmax_scale, dropout_p, causal, window_size, alibi_slopes, deterministic,
            world_size, rank, process_group, cu_seqlens_k
        )
    else:
        # Two-kernels mode: use Triton kernel for direct slicing
        dq, dk, dv = backward_two_kernels_mode(
            dout, q, k, v, out, lse,
            kv_buffer, chunk_indices_list, chunk_cu_seqlens_q_list, kv_slices,
            softmax_scale, dropout_p, causal, window_size, alibi_slopes, deterministic,
            world_size, rank, process_group, cu_seqlens_k, use_triton_kernel
        )

    return dq, dk, dv


# ============================================================================
# AutoGrad Wrapper
# ============================================================================

class ZigzagLlama3FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        use_fused_kernel_forward,
        use_fused_kernel_backward,
        n_chunks,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()

        # Execute forward
        # Use Triton kernel for two-kernels mode (not fused mode)
        use_triton = not use_fused_kernel_forward
        out, lse, chunk_info = zigzag_llama3_flash_attn_varlen_forward(
            group, q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            heads_k_stride, local_k_slice,
            softmax_scale, dropout_p, causal,
            window_size, alibi_slopes, deterministic,
            use_fused_kernel=use_fused_kernel_forward,
            n_chunks=n_chunks,
            use_triton_kernel=use_triton,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.chunk_indices_list = chunk_info['chunk_indices_list']
        ctx.chunk_cu_seqlens_q_list = chunk_info['chunk_cu_seqlens_q_list']
        ctx.kv_slices = chunk_info['kv_slices']
        ctx.use_fused_kernel_backward = use_fused_kernel_backward
        ctx.n_chunks = n_chunks
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.heads_k_stride = heads_k_stride
        ctx.local_k_slice = local_k_slice
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group

        return out if not return_softmax else (out, lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        # Use Triton kernel for two-kernels mode (not fused mode)
        use_triton = not ctx.use_fused_kernel_backward

        dq, dk, dv = zigzag_llama3_flash_attn_varlen_backward(
            ctx.group,
            dout, q, k, v, out, lse,
            cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.heads_k_stride, ctx.local_k_slice,
            ctx.softmax_scale, ctx.dropout_p, ctx.causal,
            ctx.window_size, ctx.alibi_slopes, ctx.deterministic,
            use_fused_kernel=ctx.use_fused_kernel_backward,
            n_chunks=ctx.n_chunks,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_cu_seqlens_q_list=ctx.chunk_cu_seqlens_q_list,
            kv_slices=ctx.kv_slices,
            use_triton_kernel=use_triton,
        )

        return (dq, dk, dv) + (None,) * 17


# ============================================================================
# Public API Functions
# ============================================================================

def zigzag_llama3_flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    use_fused_kernel_forward=False,
    use_fused_kernel_backward=False,
    n_chunks=2,
):
    """
    Zigzag Llama3 Flash Attention for variable-length sequences with QKV packed format.

    Args:
        qkv: [total_tokens, 3, nheads, head_dim] in zigzag interleaved format
        cu_seqlens_q: [num_seqs + 1] cumulative sequence lengths for LOCAL Q
        cu_seqlens_k: [num_seqs + 1] cumulative sequence lengths for GLOBAL K
        max_seqlen_q: Maximum sequence length in Q
        max_seqlen_k: Maximum sequence length in K
        heads_k_stride: Number of KV heads to process per iteration
        local_k_slice: Slice object for local K,V portion
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        causal: Whether to use causal attention
        window_size: Sliding window size (-1, -1) for no window
        alibi_slopes: ALiBi slopes for positional encoding
        deterministic: Whether to use deterministic backward
        return_attn_probs: Whether to return attention probabilities
        group: Process group for distributed communication
        use_fused_kernel_forward: Use fused kernel in forward pass
        use_fused_kernel_backward: Use fused kernel in backward pass
        n_chunks: Number of Q chunks (default: 2)

    Returns:
        out: [total_tokens, nheads, head_dim] output in zigzag interleaved format
    """
    return ZigzagLlama3FlashAttnVarlenFunc.apply(
        qkv[:, 0], qkv[:, 1], qkv[:, 2],
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        heads_k_stride, local_k_slice,
        dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic,
        return_attn_probs, group,
        use_fused_kernel_forward,
        use_fused_kernel_backward,
        n_chunks,
    )


def zigzag_llama3_flash_attn_varlen_kvpacked_func(
    q, kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    use_fused_kernel_forward=False,
    use_fused_kernel_backward=False,
    n_chunks=2,
):
    """
    Zigzag Llama3 Flash Attention for variable-length sequences with KV packed format.

    Args:
        q: [total_tokens, nheads, head_dim] in zigzag interleaved format
        kv: [total_tokens, 2, nheads_k, head_dim] in zigzag interleaved format
        cu_seqlens_q: [num_seqs + 1] cumulative sequence lengths for LOCAL Q
        cu_seqlens_k: [num_seqs + 1] cumulative sequence lengths for GLOBAL K
        max_seqlen_q: Maximum sequence length in Q
        max_seqlen_k: Maximum sequence length in K
        heads_k_stride: Number of KV heads to process per iteration
        local_k_slice: Slice object for local K,V portion
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        causal: Whether to use causal attention
        window_size: Sliding window size (-1, -1) for no window
        alibi_slopes: ALiBi slopes for positional encoding
        deterministic: Whether to use deterministic backward
        return_attn_probs: Whether to return attention probabilities
        group: Process group for distributed communication
        use_fused_kernel_forward: Use fused kernel in forward pass
        use_fused_kernel_backward: Use fused kernel in backward pass
        n_chunks: Number of Q chunks (default: 2)

    Returns:
        out: [total_tokens, nheads, head_dim] output in zigzag interleaved format
    """
    return ZigzagLlama3FlashAttnVarlenFunc.apply(
        q, kv[:, 0], kv[:, 1],
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        heads_k_stride, local_k_slice,
        dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic,
        return_attn_probs, group,
        use_fused_kernel_forward,
        use_fused_kernel_backward,
        n_chunks,
    )


def zigzag_llama3_flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    use_fused_kernel_forward=False,
    use_fused_kernel_backward=False,
    n_chunks=2,
):
    """
    Zigzag Llama3 Flash Attention for variable-length sequences with separate Q, K, V.

    Args:
        q: [total_tokens, nheads, head_dim] in zigzag interleaved format
        k: [total_tokens, nheads_k, head_dim] in zigzag interleaved format
        v: [total_tokens, nheads_k, head_dim] in zigzag interleaved format
        cu_seqlens_q: [num_seqs + 1] cumulative sequence lengths for LOCAL Q
        cu_seqlens_k: [num_seqs + 1] cumulative sequence lengths for GLOBAL K
        max_seqlen_q: Maximum sequence length in Q
        max_seqlen_k: Maximum sequence length in K
        heads_k_stride: Number of KV heads to process per iteration
        local_k_slice: Slice object for local K,V portion
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        causal: Whether to use causal attention
        window_size: Sliding window size (-1, -1) for no window
        alibi_slopes: ALiBi slopes for positional encoding
        deterministic: Whether to use deterministic backward
        return_attn_probs: Whether to return attention probabilities
        group: Process group for distributed communication
        use_fused_kernel_forward: Use fused kernel in forward pass
        use_fused_kernel_backward: Use fused kernel in backward pass
        n_chunks: Number of Q chunks (default: 2)

    Returns:
        out: [total_tokens, nheads, head_dim] output in zigzag interleaved format
    """
    return ZigzagLlama3FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        heads_k_stride, local_k_slice,
        dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic,
        return_attn_probs, group,
        use_fused_kernel_forward,
        use_fused_kernel_backward,
        n_chunks,
    )

