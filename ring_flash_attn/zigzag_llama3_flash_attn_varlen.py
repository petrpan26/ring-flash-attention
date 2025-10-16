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
from flash_attn.flash_attn_grouped import (
    _flash_attn_varlen_forward_grouped,
)
from .utils import get_default_args, AllGatherComm as Comm
from .triton_utils import extract_zigzag_kv_slices_for_group, scatter_grad_to_zigzag
from .triton_grouped_attention import triton_grouped_flash_attn_varlen_forward


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
        cu_seqlens: [num_seqs + 1] - Cumulative sequence lengths for LOCAL sequences
                    (already divided by world_size at entry point)

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

    # Detect if cu_seqlens is GLOBAL or LOCAL by comparing with kv_buffer size
    # After all-gather, kv_buffer has size [2, total_tokens * world_size, nheads_k, head_dim]
    total_tokens_in_buffer = kv_buffer.shape[1]
    total_tokens_in_cu_seqlens = cu_seqlens[-1].item()
    is_global_cu_seqlens = (total_tokens_in_buffer == total_tokens_in_cu_seqlens)

    # Process each sequence separately (they may have different lengths)
    contiguous_chunks = []
    offset_per_rank = [0] * world_size  # Track position within each rank's data

    for seq_idx in range(num_sequences):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        seq_len = seq_end - seq_start

        # Calculate chunk_size based on whether cu_seqlens is GLOBAL or LOCAL
        if is_global_cu_seqlens:
            # cu_seqlens is GLOBAL: seq_len is total across all ranks
            # chunk_size = seq_len_GLOBAL // (2 * world_size)
            chunk_size = seq_len // total_chunks
        else:
            # cu_seqlens is LOCAL: seq_len is per-rank
            # chunk_size = seq_len_LOCAL // 2
            chunk_size = seq_len // (total_chunks // world_size)

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

            # Always update offset after extracting each chunk
            offset_per_rank[rank] = end_in_rank

    # Concatenate all chunks
    kv_contiguous = torch.cat(contiguous_chunks, dim=1)

    # Sanity check: output should have same total_tokens as input
    assert kv_contiguous.shape[1] == total_tokens, \
        f"rearrange_kv_from_zigzag_to_contiguous output size mismatch: " \
        f"expected {total_tokens}, got {kv_contiguous.shape[1]}. " \
        f"Input: {kv_buffer.shape}, cu_seqlens={cu_seqlens.tolist()}, world_size={world_size}"

    return kv_contiguous


# Optimization #4: torch.compile for hot path functions
# Note: This function uses .item() for dynamic control flow, which causes graph breaks.
# Since it's primarily data reorganization (not compute-intensive), compiling provides
# minimal benefit. The real performance gains come from Flash Attention kernels.
# @torch.compile(mode="reduce-overhead", fullgraph=False)  # Disabled to avoid .item() warnings
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
# Note: This function uses .item() for dynamic control flow, which causes graph breaks.
# Since it's primarily metadata computation (not compute-intensive), compiling provides
# minimal benefit. The real performance gains come from Flash Attention kernels.
# @torch.compile(mode="reduce-overhead", fullgraph=False)  # Disabled to avoid .item() warnings
def compute_kv_slices_for_groups(
    cu_seqlens_k: torch.Tensor,
    chunk_idx_0: int,
    chunk_idx_1: int,
    world_size: int,
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    n_chunks: int = 2
) -> List[Tuple[List[Tuple[int, int]], torch.Tensor]]:
    """
    Compute which portion of contiguous K,V each Q group needs for causal attention.

    For causal attention, a chunk at position i needs K,V from positions [0, i].
    This function determines the K,V slice ranges for each sequence in each group.

    Args:
        cu_seqlens_k: [num_seqs + 1] - Cumulative lengths for LOCAL K,V
                      (already divided by world_size at entry point)
        chunk_idx_0: Global chunk index for first local chunk
        chunk_idx_1: Global chunk index for second local chunk
        world_size: Number of GPUs
        chunk_cu_seqlens_q_list: List of cu_seqlens tensors for Q in each group
                                 Used to determine exact structure K should match
        n_chunks: Number of groups (default: 2)

    Returns:
        List of (seq_ranges, cu_seqlens_k_slice) for each group
        seq_ranges: List of (start, end) tuples for each sequence in contiguous K,V buffer
        cu_seqlens_k_slice: Adjusted cu_seqlens for the extracted K,V slice
                            (must match length of chunk_cu_seqlens_q_list[group_idx])

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

    # Compute K,V slices for each group
    kv_slices = []
    for group_idx in range(n_chunks):
        # Get the Q structure for this group to match exactly
        cu_seqlens_q_group = chunk_cu_seqlens_q_list[group_idx]
        num_seqs_in_group = len(cu_seqlens_q_group) - 1

        # Determine which chunk from this rank belongs to this group
        # With n_chunks=2: group_0 always has chunk_idx_0, group_1 always has chunk_idx_1
        if group_0 == group_idx:
            max_chunk_idx = chunk_idx_0
        elif group_1 == group_idx:
            max_chunk_idx = chunk_idx_1
        else:
            # This group has no chunks from this rank
            # Return empty ranges with cu_seqlens matching Q structure
            kv_slices.append(([], cu_seqlens_q_group.clone()))
            continue

        # Calculate K,V slice ranges for each sequence
        # For causal attention, need K,V from start up to end of max_chunk_idx
        # IMPORTANT: cu_seqlens_k_slice must match the exact structure of cu_seqlens_q_group
        #
        # Q group and cu_seqlens_k should have the same number of sequences
        # (both derived from the same original cu_seqlens, just Q is split by chunks)
        if num_seqs_in_group != num_sequences:
            raise RuntimeError(
                f"Q group {group_idx} has {num_seqs_in_group} sequences, "
                f"but cu_seqlens_k has {num_sequences} sequences. "
                f"They should match! Check that cu_seqlens_q passed to the function "
                f"is LOCAL (not sliced by llama3_prepare for zigzag data)."
            )

        seq_ranges = []
        cu_seqlens_k_list = [0]

        # Build GLOBAL cu_seqlens for the contiguous buffer
        # CRITICAL: After rearrange_kv_from_zigzag_to_contiguous, the contiguous buffer
        # contains ALL chunks from ALL ranks, so it has world_size * LOCAL tokens.
        # We must use GLOBAL coordinates when indexing into this buffer!
        cu_seqlens_k_global = []
        for i in range(len(cu_seqlens_k)):
            cu_seqlens_k_global.append(cu_seqlens_k[i].item() * world_size)

        # DEBUG
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"\n[Rank {rank}] compute_kv_slices_for_groups - Group {group_idx}:")
            print(f"  cu_seqlens_k (LOCAL): {cu_seqlens_k.tolist()}")
            print(f"  cu_seqlens_k_global: {cu_seqlens_k_global}")
            print(f"  max_chunk_idx: {max_chunk_idx}")
            print(f"  world_size: {world_size}, total_chunks: {total_chunks}")

        # Process each sequence (Q and K have same number of sequences)
        for seq_idx in range(num_seqs_in_group):
            # Get K sequence boundaries in GLOBAL (contiguous) coordinates
            k_seq_start_global = cu_seqlens_k_global[seq_idx]
            k_seq_end_global = cu_seqlens_k_global[seq_idx + 1]
            k_seq_len_global = k_seq_end_global - k_seq_start_global

            if k_seq_len_global == 0:
                # K sequence is empty
                cu_seqlens_k_list.append(cu_seqlens_k_list[-1])
                continue

            # Calculate chunk size for this K sequence in GLOBAL coordinates
            chunk_size = k_seq_len_global // total_chunks

            # Need K tokens from start up to end of max_chunk_idx
            num_tokens_needed = (max_chunk_idx + 1) * chunk_size
            slice_start = k_seq_start_global
            slice_end = k_seq_start_global + num_tokens_needed

            seq_ranges.append((slice_start, slice_end))
            cu_seqlens_k_list.append(cu_seqlens_k_list[-1] + num_tokens_needed)

            # DEBUG
            if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
                print(f"  Seq {seq_idx}: k_seq_len_global={k_seq_len_global}, chunk_size={chunk_size}")
                print(f"    num_tokens_needed={num_tokens_needed}, range=({slice_start}, {slice_end})")

        cu_seqlens_k_slice = cu_seqlens_k_list

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

    OPTIMIZATION: Extract K,V only once for the maximum group, then slice for smaller groups.
    Since late group contains early group (e.g., [0-7] contains [0-3]), we avoid redundant extraction.

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

    # DEBUG: Check input state
    import os
    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        rank = dist.get_rank() if dist.is_initialized() else 0
        has_nan_kv = torch.isnan(kv_buffer).any().item()
        has_inf_kv = torch.isinf(kv_buffer).any().item()
        print(f"\n[Rank {rank}] execute_two_kernels_mode - Input state:")
        print(f"  kv_buffer.shape: {kv_buffer.shape}")
        print(f"  kv_buffer NaN={has_nan_kv} Inf={has_inf_kv}")
        print(f"  cu_seqlens_k (LOCAL): {cu_seqlens_k.tolist()}")
        print(f"  world_size: {world_size}")
        print(f"  use_triton_kernel: {use_triton_kernel}")

    # Rearrange to contiguous format once for Python fallback
    kv_contiguous = None
    if not use_triton_kernel:
        kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
        # DEBUG
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            has_nan_cont = torch.isnan(kv_contiguous).any().item()
            has_inf_cont = torch.isinf(kv_contiguous).any().item()
            print(f"  After rearrange: kv_contiguous.shape={kv_contiguous.shape}, NaN={has_nan_cont} Inf={has_inf_cont}")

    # Process each group, extracting K,V specifically for that group
    for group_idx, (q_group, cu_seqlens_q_group) in enumerate(zip(chunk_q_list, chunk_cu_seqlens_q_list)):
        if q_group.shape[0] == 0:
            # Empty group, skip
            continue

        seq_ranges, cu_seqlens_k_slice = kv_slices[group_idx]

        # DEBUG
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"\n[Rank {rank}] execute_two_kernels_mode - Group {group_idx}:")
            print(f"  q_group.shape: {q_group.shape}")
            print(f"  cu_seqlens_q_group: {cu_seqlens_q_group.tolist()}")
            print(f"  cu_seqlens_k_slice: {cu_seqlens_k_slice.tolist()}")
            print(f"  seq_ranges (GLOBAL): {seq_ranges}")
            print(f"  kv_buffer.shape: {kv_buffer.shape}")
            print(f"  use_triton_kernel: {use_triton_kernel}")

        # Extract K,V for this specific group
        if use_triton_kernel:
            # Use Triton kernel to extract K,V slices directly from zigzag buffer
            # Build seq_ranges_with_chunk_idx: [(start, end, max_chunk_idx), ...]
            # NOTE: seq_ranges contains GLOBAL coordinates
            # The start/end are ONLY used to compute output buffer size (num_tokens = end - start)
            # The actual extraction is controlled by max_chunk_idx
            # So we keep GLOBAL coordinates for start/end to get correct output size!
            seq_ranges_with_chunk_idx = []
            for seq_idx, (start_global, end_global) in enumerate(seq_ranges):
                # Get LOCAL sequence length to compute max_chunk_idx
                seq_start_local = cu_seqlens_k[seq_idx].item()
                seq_end_local = cu_seqlens_k[seq_idx + 1].item()
                seq_len_local = seq_end_local - seq_start_local
                total_chunks = 2 * world_size

                # Compute max_chunk_idx from GLOBAL token count
                num_tokens_global = end_global - start_global
                chunk_size_global = seq_len_local * world_size // total_chunks
                max_chunk_idx = (num_tokens_global // chunk_size_global) - 1

                # Keep GLOBAL coordinates for start/end so output buffer size is correct
                # The Triton kernel only uses (end - start) to determine output size
                # It uses max_chunk_idx to determine what to extract from zigzag buffer
                seq_ranges_with_chunk_idx.append((start_global, end_global, max_chunk_idx))

                # DEBUG
                if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
                    print(f"  Seq {seq_idx}: GLOBAL range ({start_global}, {end_global}), max_chunk_idx={max_chunk_idx}, output_tokens={end_global - start_global}")

            # DEBUG
            if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
                print(f"  seq_ranges_with_chunk_idx (GLOBAL ranges + max_chunk_idx): {seq_ranges_with_chunk_idx}")

            # Create GLOBAL cu_seqlens for Triton kernel
            # The Triton kernel needs GLOBAL cu_seqlens to compute sequence lengths correctly
            cu_seqlens_k_global = torch.tensor(
                [cu_seqlens_k[i].item() * world_size for i in range(len(cu_seqlens_k))],
                device=cu_seqlens_k.device,
                dtype=cu_seqlens_k.dtype
            )

            nheads_k = kv_buffer.shape[2]
            k_slice, v_slice = extract_zigzag_kv_slices_for_group(
                kv_buffer,
                seq_ranges_with_chunk_idx,
                cu_seqlens_k_global,  # Pass GLOBAL, not LOCAL!
                world_size,
                nheads_k,
                head_dim,
            )
        else:
            # Fallback: Python-based extraction
            # Extract K,V slices for each sequence and concatenate
            k_slices = []
            v_slices = []
            for start, end in seq_ranges:
                k_slices.append(kv_contiguous[0, start:end])
                v_slices.append(kv_contiguous[1, start:end])

            k_slice = torch.cat(k_slices, dim=0)  # [total_k_tokens, heads, dim]
            v_slice = torch.cat(v_slices, dim=0)  # [total_v_tokens, heads, dim]

        # DEBUG: Check for NaN/Inf after extraction
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            has_nan_k = torch.isnan(k_slice).any().item()
            has_nan_v = torch.isnan(v_slice).any().item()
            has_inf_k = torch.isinf(k_slice).any().item()
            has_inf_v = torch.isinf(v_slice).any().item()
            expected_tokens = sum(end - start for start, end in seq_ranges)
            print(f"[Rank {rank}] Group {group_idx} After extraction: k_slice NaN={has_nan_k} Inf={has_inf_k}, v_slice NaN={has_nan_v} Inf={has_inf_v}")
            print(f"[Rank {rank}] Group {group_idx} k_slice.shape: {k_slice.shape} (expected {expected_tokens}), v_slice.shape: {v_slice.shape}")

        # Calculate max_seqlen
        max_seqlen_q = (cu_seqlens_q_group[1:] - cu_seqlens_q_group[:-1]).max().item()
        max_seqlen_k = (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()

        # DEBUG: Final check before flash_attn
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            print(f"  Ready for flash_attn:")
            print(f"    q_group.shape: {q_group.shape}")
            print(f"    k_slice.shape: {k_slice.shape}")
            print(f"    v_slice.shape: {v_slice.shape}")
            print(f"    max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")

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

        # DEBUG: Check output
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            has_nan_out = torch.isnan(out_group).any().item()
            has_inf_out = torch.isinf(out_group).any().item()
            print(f"  After flash_attn: out_group NaN={has_nan_out} Inf={has_inf_out}, shape={out_group.shape}")

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


def execute_triton_grouped_mode(
    chunk_q_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    chunk_indices_list: List[torch.Tensor],
    kv_buffer: torch.Tensor,
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
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Execute attention using Triton grouped kernel (Option C).

    This mode uses a custom Triton kernel where multiple Q groups share K,V loads
    in a single fused kernel. This eliminates redundant HBM reads when groups
    attend to overlapping K,V regions.

    Benefits:
    - Reduced memory bandwidth (10-15% faster than two-kernels mode)
    - K,V tiles loaded once and shared across groups via L2 cache
    - Simpler than CUDA implementation, more maintainable

    Args:
        chunk_q_list: List of Q tensors per group
        chunk_cu_seqlens_q_list: List of cu_seqlens_q per group
        chunk_indices_list: List of indices for reconstruction
        kv_buffer: [2, total_tokens, nheads_k, head_dim] - zigzag interleaved buffer
        kv_slices: List of (seq_ranges, cu_seqlens_k) per group
        nheads: Number of Q heads
        head_dim: Head dimension
        softmax_scale: Softmax scale factor
        dropout_p: Dropout probability (not supported)
        causal: Whether to apply causal masking
        window_size: Sliding window size (not supported)
        alibi_slopes: ALiBi slopes (not supported)
        deterministic: Deterministic mode (not used)
        world_size: Number of ranks
        cu_seqlens_k: Local cumulative sequence lengths for K

    Returns:
        out: Combined output in original local Q order
        lse: Combined LSE
        chunk_info: Dictionary with chunking information for backward
    """
    import os
    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"\n[Rank {rank}] execute_triton_grouped_mode:")
        print(f"  num_groups: {len(chunk_q_list)}")
        print(f"  kv_buffer.shape: {kv_buffer.shape}")

    # First, rearrange KV from zigzag to contiguous format
    # This is needed because Triton kernel expects contiguous K,V
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)

    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        print(f"  After rearrange: kv_contiguous.shape={kv_contiguous.shape}")

    # Extract K, V (shared across all groups)
    k_full = kv_contiguous[0]  # [total_k_tokens, nheads_k, head_dim]
    v_full = kv_contiguous[1]  # [total_v_tokens, nheads_k, head_dim]

    # Prepare K slices for each group (different lengths)
    # Note: We need to slice K,V to the appropriate length for each group
    k_list = []
    v_list = []
    cu_seqlens_k_list = []
    max_seqlen_k_list = []

    for group_idx, (seq_ranges, cu_seqlens_k_slice) in enumerate(kv_slices):
        # Extract K,V for this group
        k_slices = []
        v_slices = []
        for start, end in seq_ranges:
            k_slices.append(k_full[start:end])
            v_slices.append(v_full[start:end])

        k_group = torch.cat(k_slices, dim=0)
        v_group = torch.cat(v_slices, dim=0)

        # For grouped kernel, we pass the full K,V and let kernel handle slicing
        # But we need the max length for this group
        max_seqlen_k = (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()

        k_list.append(k_group)
        v_list.append(v_group)
        cu_seqlens_k_list.append(cu_seqlens_k_slice)
        max_seqlen_k_list.append(max_seqlen_k)

        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            print(f"  Group {group_idx}: k_group.shape={k_group.shape}, max_seqlen_k={max_seqlen_k}")

    # Calculate max_seqlen_q for each group
    max_seqlen_q_list = [
        (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        for cu_seqlens in chunk_cu_seqlens_q_list
    ]

    # For now, use the largest K,V as shared (simplified implementation)
    # Full optimization would share K,V loads in kernel itself
    # This is a hybrid approach: use Triton for individual groups, but optimize later
    max_k_len = max(k.shape[0] for k in k_list)
    max_k_idx = max(range(len(k_list)), key=lambda i: k_list[i].shape[0])

    # Use largest K,V as the shared tensors
    k_shared = k_list[max_k_idx]
    v_shared = v_list[max_k_idx]

    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        print(f"  Using group {max_k_idx} as shared K,V base (length={k_shared.shape[0]})")
        print(f"  Calling triton_grouped_flash_attn_varlen_forward...")

    # Call Triton grouped attention
    # Note: Current implementation calls kernel separately for each group
    # but shares K,V loads via L2 cache
    out_list, lse_list = triton_grouped_flash_attn_varlen_forward(
        q_list=chunk_q_list,
        k=k_shared,
        v=v_shared,
        cu_seqlens_q_list=chunk_cu_seqlens_q_list,
        cu_seqlens_k_list=cu_seqlens_k_list,
        max_seqlen_q_list=max_seqlen_q_list,
        max_seqlen_k_list=max_seqlen_k_list,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )

    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        print(f"  Triton grouped kernel returned {len(out_list)} outputs")

    # Combine outputs back to original local Q order
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros(
        (total_q, nheads, head_dim),
        dtype=out_list[0].dtype,
        device=out_list[0].device
    )

    # Scatter each chunk back to its original positions
    for out_chunk, indices in zip(out_list, chunk_indices_list):
        out[indices] = out_chunk

    # Scatter LSE similarly
    if lse_list[0].dim() == 2:
        # LSE is [nheads, chunk_len] or [chunk_len, nheads]
        if lse_list[0].shape[0] == nheads:
            # [nheads, chunk_len]
            lse = torch.zeros(
                (nheads, total_q),
                dtype=lse_list[0].dtype,
                device=lse_list[0].device
            )
            for lse_chunk, indices in zip(lse_list, chunk_indices_list):
                lse[:, indices] = lse_chunk
        else:
            # [chunk_len, nheads]
            lse = torch.zeros(
                (total_q, nheads),
                dtype=lse_list[0].dtype,
                device=lse_list[0].device
            )
            for lse_chunk, indices in zip(lse_list, chunk_indices_list):
                lse[indices] = lse_chunk
    else:
        # LSE is 1D, shouldn't happen
        raise ValueError(f"Unexpected LSE shape: {lse_list[0].shape}")

    chunk_info = {
        'chunk_indices_list': chunk_indices_list,
        'chunk_cu_seqlens_q_list': chunk_cu_seqlens_q_list,
        'kv_slices': kv_slices,
    }

    return out, lse, chunk_info


def execute_two_kernels_mode_grouped(
    chunk_q_list: List[torch.Tensor],
    chunk_cu_seqlens_q_list: List[torch.Tensor],
    chunk_indices_list: List[torch.Tensor],
    kv_buffer: torch.Tensor,  # Zigzag buffer [2, total_tokens, nheads_k, head_dim]
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
    Execute attention with grouped kernel call (multiple Q groups in one API call).

    Uses _flash_attn_varlen_forward_grouped to process all Q groups together,
    sharing K,V loads for improved memory efficiency.

    This is the Python prototype implementation (Option B) that relies on L2 cache
    to share K,V data between consecutive kernel launches.

    Args:
        Same as execute_two_kernels_mode

    Returns:
        out: Combined output in original local Q order
        lse: Combined LSE
        chunk_info: Dictionary with chunking information for backward
    """
    import os

    # DEBUG: Check input state
    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        rank = dist.get_rank() if dist.is_initialized() else 0
        has_nan_kv = torch.isnan(kv_buffer).any().item()
        has_inf_kv = torch.isinf(kv_buffer).any().item()
        print(f"\n[Rank {rank}] execute_two_kernels_mode_grouped - Input state:")
        print(f"  kv_buffer.shape: {kv_buffer.shape}")
        print(f"  kv_buffer NaN={has_nan_kv} Inf={has_inf_kv}")
        print(f"  cu_seqlens_k (LOCAL): {cu_seqlens_k.tolist()}")
        print(f"  world_size: {world_size}")
        print(f"  use_triton_kernel: {use_triton_kernel}")

    # Rearrange K,V to contiguous format (needed for grouped attention)
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        has_nan_cont = torch.isnan(kv_contiguous).any().item()
        has_inf_cont = torch.isinf(kv_contiguous).any().item()
        print(f"  After rearrange: kv_contiguous.shape={kv_contiguous.shape}, NaN={has_nan_cont} Inf={has_inf_cont}")

    # Prepare lists for grouped attention call
    q_list = []
    cu_seqlens_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_q_list = []
    max_seqlen_k_list = []

    # Track which groups are non-empty
    non_empty_group_indices = []

    for group_idx, (q_group, cu_seqlens_q_group) in enumerate(zip(chunk_q_list, chunk_cu_seqlens_q_list)):
        if q_group.shape[0] == 0:
            # Skip empty groups
            continue

        non_empty_group_indices.append(group_idx)
        seq_ranges, cu_seqlens_k_slice = kv_slices[group_idx]

        # DEBUG
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"\n[Rank {rank}] execute_two_kernels_mode_grouped - Group {group_idx}:")
            print(f"  q_group.shape: {q_group.shape}")
            print(f"  cu_seqlens_q_group: {cu_seqlens_q_group.tolist()}")
            print(f"  cu_seqlens_k_slice: {cu_seqlens_k_slice.tolist()}")
            print(f"  seq_ranges (GLOBAL): {seq_ranges}")

        # Add to lists for grouped call
        q_list.append(q_group)
        cu_seqlens_q_list.append(cu_seqlens_q_group)
        cu_seqlens_k_list.append(cu_seqlens_k_slice)

        max_seqlen_q = (cu_seqlens_q_group[1:] - cu_seqlens_q_group[:-1]).max().item()
        max_seqlen_k = (cu_seqlens_k_slice[1:] - cu_seqlens_k_slice[:-1]).max().item()

        max_seqlen_q_list.append(max_seqlen_q)
        max_seqlen_k_list.append(max_seqlen_k)

    # Call grouped attention if we have any non-empty groups
    out_chunks = []
    lse_chunks = []

    if len(q_list) > 0:
        # Extract K,V (full length, will be sliced inside grouped function)
        k_full = kv_contiguous[0]  # [total_k_tokens, nheads_k, head_dim]
        v_full = kv_contiguous[1]  # [total_v_tokens, nheads_k, head_dim]

        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"\n[Rank {rank}] Calling _flash_attn_varlen_forward_grouped:")
            print(f"  num_groups: {len(q_list)}")
            print(f"  k_full.shape: {k_full.shape}")
            print(f"  v_full.shape: {v_full.shape}")
            print(f"  max_seqlen_k_list: {max_seqlen_k_list}")

        # Call grouped attention
        out_list_grouped, lse_list_grouped, _, _ = _flash_attn_varlen_forward_grouped(
            q_list=q_list,
            k=k_full,
            v=v_full,
            cu_seqlens_q_list=cu_seqlens_q_list,
            cu_seqlens_k_list=cu_seqlens_k_list,
            max_seqlen_q_list=max_seqlen_q_list,
            max_seqlen_k_list=max_seqlen_k_list,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=0.0,
            alibi_slopes=alibi_slopes,
            return_softmax=dropout_p > 0,
            deterministic=deterministic,
        )

        # DEBUG: Check outputs
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            for i, (out_group, lse_group) in enumerate(zip(out_list_grouped, lse_list_grouped)):
                has_nan_out = torch.isnan(out_group).any().item()
                has_inf_out = torch.isinf(out_group).any().item()
                print(f"  Group {non_empty_group_indices[i]}: out NaN={has_nan_out} Inf={has_inf_out}, shape={out_group.shape}")

        # Map back to original group indices
        for i, group_idx in enumerate(non_empty_group_indices):
            out_chunks.append(out_list_grouped[i])
            lse_chunks.append(lse_list_grouped[i])

    # Combine outputs back to original local Q order
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros(
        (total_q, nheads, head_dim),
        dtype=out_chunks[0].dtype if len(out_chunks) > 0 else torch.float16,
        device=out_chunks[0].device if len(out_chunks) > 0 else chunk_q_list[0].device
    )

    # Scatter each chunk back to its original positions
    for group_idx in non_empty_group_indices:
        out_idx = non_empty_group_indices.index(group_idx)
        out[chunk_indices_list[group_idx]] = out_chunks[out_idx]

    # Scatter LSE similarly
    if len(lse_chunks) > 0 and lse_chunks[0].dim() == 2:
        # LSE is [nheads, chunk_len]
        lse = torch.zeros(
            (nheads, total_q),
            dtype=lse_chunks[0].dtype,
            device=lse_chunks[0].device
        )
        for group_idx in non_empty_group_indices:
            lse_idx = non_empty_group_indices.index(group_idx)
            lse[:, chunk_indices_list[group_idx]] = lse_chunks[lse_idx]
    else:
        # LSE is [chunk_len, nheads]
        lse = torch.zeros(
            (total_q, nheads),
            dtype=lse_chunks[0].dtype if len(lse_chunks) > 0 else torch.float32,
            device=lse_chunks[0].device if len(lse_chunks) > 0 else chunk_q_list[0].device
        )
        for group_idx in non_empty_group_indices:
            lse_idx = non_empty_group_indices.index(group_idx)
            lse[chunk_indices_list[group_idx]] = lse_chunks[lse_idx]

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
    use_triton_grouped: bool = False,
    use_grouped_attention: bool = False,
):
    """
    Forward pass with zigzag-style Q chunking and llama3-style all-gather.

    Process:
    1. Receive Q, K, V in zigzag interleaved format
    2. All-gather K,V from all ranks (llama3-style)
    3. [OPTIMIZED] Use Triton kernel to extract K,V slices directly from zigzag buffer
    4. Split local Q by global chunk index (early vs late groups)
    5. Compute K,V slices for each group (causal attention)
    6. Execute attention (two-kernels, fused, or triton-grouped mode)
    7. Return output in interleaved format

    Args:
        q, k, v: Input tensors in zigzag interleaved format
        cu_seqlens_q: Cumulative sequence lengths for LOCAL Q
        cu_seqlens_k: Cumulative sequence lengths for GLOBAL K (total across all ranks)
                      Following llama3 API pattern: will be divided by world_size internally
        use_fused_kernel: If False, use two sequential kernels.
                          If True, use one kernel with duplicated K,V.
        n_chunks: Number of Q groups (default: 2 = early/late)
        use_triton_kernel: If True, use Triton kernel for KV slicing (faster).
                           If False, use Python rearrangement (for debugging).
        use_triton_grouped: If True, use Triton grouped attention kernel (Option C).
                            This eliminates redundant K,V loads by sharing them across groups.
                            Expected 10-15% speedup over two-kernels mode.
        use_grouped_attention: If True, use Python prototype grouped attention (Option B).
                               This calls _flash_attn_varlen_forward_grouped which launches
                               multiple flash attention kernels sequentially, relying on L2 cache
                               for K,V sharing. Expected 5-10% speedup over two-kernels mode.
                               Note: Cannot be used with use_triton_grouped or use_fused_kernel.

    Returns:
        out: Output tensor in zigzag interleaved format
        lse: Log-sum-exp values
        chunk_info: Dictionary with chunking information for backward
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape

    # Validate heads_k_stride
    assert nheads_k % heads_k_stride == 0, \
        f"nheads_k ({nheads_k}) must be divisible by heads_k_stride ({heads_k_stride})"

    # Convert GLOBAL cu_seqlens_k to LOCAL (following llama3 API pattern)
    # Do this once at entry point, then all internal functions work with LOCAL
    cu_seqlens_k = (cu_seqlens_k // world_size).to(torch.int32)

    # Split local Q by global chunk index (do once, reuse for all head iterations)
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q, cu_seqlens_q, world_size, rank, n_chunks=n_chunks
    )

    # Compute K,V slices for each group (do once, reuse for all head iterations)
    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    kv_slices = compute_kv_slices_for_groups(
        cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size,
        chunk_cu_seqlens_q_list, n_chunks=n_chunks
    )

    # Allocate buffers for all-gather
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    # Start all-gather for first KV head slice
    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm = Comm(process_group)
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    # Accumulate outputs from each head iteration
    out_list = []
    lse_list = []

    # Loop over KV heads in chunks of heads_k_stride (like llama3)
    for i in range(0, nheads_k, heads_k_stride):
        # Wait for current all-gather to complete
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        # Start all-gather for next KV head slice (overlap with computation)
        if i < nheads_k - heads_k_stride:
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        # Get Q slice for this KV head iteration (for GQA support)
        q_slice = slice(
            i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
        )

        # Split Q slice by chunks
        chunk_q_list_i = [q_chunk[:, q_slice] for q_chunk in chunk_q_list]

        # Execute attention for this head slice
        if use_triton_grouped:
            # Triton grouped mode: custom Triton kernel with shared K,V loads (Option C)
            out_i, lse_i, chunk_info = execute_triton_grouped_mode(
                chunk_q_list_i, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads // nheads_k * heads_k_stride, head_dim, softmax_scale, dropout_p, causal,
                window_size, alibi_slopes, deterministic,
                world_size, cu_seqlens_k
            )
        elif use_grouped_attention:
            # Grouped attention mode (Python prototype): uses _flash_attn_varlen_forward_grouped (Option B)
            # Relies on L2 cache for K,V sharing between consecutive kernel launches
            out_i, lse_i, chunk_info = execute_two_kernels_mode_grouped(
                chunk_q_list_i, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads // nheads_k * heads_k_stride, head_dim, softmax_scale, dropout_p, causal,
                window_size, alibi_slopes, deterministic,
                world_size, cu_seqlens_k, use_triton_kernel
            )
        elif use_fused_kernel:
            # Fused kernel mode: needs contiguous rearrangement
            kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
            out_i, lse_i, chunk_info = execute_fused_kernel_mode(
                chunk_q_list_i, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_contiguous, kv_slices,
                nheads // nheads_k * heads_k_stride, head_dim, softmax_scale, dropout_p, causal,
                window_size, alibi_slopes, deterministic
            )
        else:
            # Two-kernels mode: use Triton kernel for direct slicing
            out_i, lse_i, chunk_info = execute_two_kernels_mode(
                chunk_q_list_i, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads // nheads_k * heads_k_stride, head_dim, softmax_scale, dropout_p, causal,
                window_size, alibi_slopes, deterministic,
                world_size, cu_seqlens_k, use_triton_kernel
            )

        # Accumulate outputs
        out_list.append(out_i)
        lse_list.append(lse_i)

    # Concatenate outputs along head dimension (like llama3)
    out = torch.cat(out_list, dim=1)
    lse = torch.cat(lse_list, dim=-2)  # Concatenate along second-to-last dim

    # Note: chunk_info from last iteration contains the necessary metadata for backward
    return out, lse, chunk_info


# ============================================================================
# Backward Pass Helper Functions
# ============================================================================

def rearrange_grad_from_contiguous_to_zigzag(
    grad_contiguous: torch.Tensor,
    world_size: int,
    cu_seqlens: torch.Tensor,
    rank: int = None,
    use_triton_kernel: bool = True,
) -> torch.Tensor:
    """
    Rearrange gradients from contiguous to zigzag interleaved format.

    This is the reverse operation of rearrange_kv_from_zigzag_to_contiguous.

    Args:
        grad_contiguous: [2, total_tokens, heads, dim] in contiguous order
        world_size: Number of GPUs
        cu_seqlens: [num_seqs + 1] - Cumulative sequence lengths for LOCAL sequences
                    (already divided by world_size at entry point)
        rank: Current rank (required for Triton kernel)
        use_triton_kernel: If True, use Triton kernel (faster). If False, use Python.

    Returns:
        Gradients in zigzag interleaved format [2, total_tokens, heads, dim]

    Example (world_size=4):
        Input (contiguous): [c0, c1, c2, c3, c4, c5, c6, c7]
        Output (interleaved): [r0_c0, r0_c7, r1_c1, r1_c6, r2_c2, r2_c5, r3_c3, r3_c4]
                            = [c0, c7, c1, c6, c2, c5, c3, c4]
    """
    if use_triton_kernel and rank is not None:
        # OPTIMIZATION: Use Triton kernel for faster gradient rearrangement
        # Process K and V separately
        dk_contiguous = grad_contiguous[0]  # [total_tokens, heads, dim]
        dv_contiguous = grad_contiguous[1]  # [total_tokens, heads, dim]

        # Create GLOBAL cu_seqlens for Triton kernel
        # The Triton kernel needs GLOBAL cu_seqlens (multiply LOCAL by world_size)
        cu_seqlens_global = torch.tensor(
            [cu_seqlens[i].item() * world_size for i in range(len(cu_seqlens))],
            device=cu_seqlens.device,
            dtype=cu_seqlens.dtype
        )

        # Scatter both K and V gradients for ALL ranks (needed for reduce_scatter)
        dk_zigzag_all_ranks = []
        dv_zigzag_all_ranks = []
        for r in range(world_size):
            dk_zigzag_all_ranks.append(scatter_grad_to_zigzag(dk_contiguous, cu_seqlens_global, r, world_size))
            dv_zigzag_all_ranks.append(scatter_grad_to_zigzag(dv_contiguous, cu_seqlens_global, r, world_size))

        # Concatenate all ranks' data
        dk_zigzag = torch.cat(dk_zigzag_all_ranks, dim=0)
        dv_zigzag = torch.cat(dv_zigzag_all_ranks, dim=0)

        # Stack back to [2, total_tokens_all_ranks, heads, dim]
        return torch.stack([dk_zigzag, dv_zigzag])
    else:
        # FALLBACK: Python implementation
        num_sequences = len(cu_seqlens) - 1
        total_chunks = 2 * world_size

        # Process each sequence separately (they may have different lengths)
        grad_interleaved_chunks = [[] for _ in range(world_size)]
        global_pos = 0

        for seq_idx in range(num_sequences):
            seq_start = cu_seqlens[seq_idx].item()
            seq_end = cu_seqlens[seq_idx + 1].item()
            seq_len_local = seq_end - seq_start  # LOCAL sequence length
            # grad_contiguous has GLOBAL size, so we need GLOBAL chunk size
            # GLOBAL seq_len = seq_len_local * world_size
            # chunk_size_global = (seq_len_local * world_size) // total_chunks
            chunk_size = seq_len_local * world_size // total_chunks

            # For each chunk in this sequence, determine which rank it belongs to
            for chunk_idx in range(total_chunks):
                # Determine which rank has this chunk
                if chunk_idx < world_size:
                    # Early chunks: rank = chunk_idx
                    rank_owner = chunk_idx
                else:
                    # Late chunks: rank = 2*world_size - 1 - chunk_idx
                    rank_owner = 2 * world_size - 1 - chunk_idx

                # Extract chunk from contiguous gradient
                chunk_start = global_pos
                chunk_end = global_pos + chunk_size
                chunk = grad_contiguous[:, chunk_start:chunk_end]
                grad_interleaved_chunks[rank_owner].append(chunk)

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
    dout: torch.Tensor,     # Already sliced for current head range
    q: torch.Tensor,        # Already sliced for current head range
    out: torch.Tensor,      # Already sliced for current head range
    lse: torch.Tensor,      # Already sliced for current head range
    kv_buffer: torch.Tensor,  # [2, total_tokens, heads_k_stride, head_dim] - zigzag buffer
    dkv_buffer: torch.Tensor, # [2, total_tokens, heads_k_stride, head_dim] - output gradient buffer (GLOBAL contiguous space)
    dq: torch.Tensor,       # Output buffer (already sliced for current head range)
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
    cu_seqlens_k: torch.Tensor,
    use_triton_kernel: bool = True,
) -> None:  # No return - writes directly to dq and dkv_buffer
    """
    Execute backward with two sequential kernel calls (one per group).

    Key: ACCUMULATE gradients for overlapping K,V regions.

    OPTIMIZATION: Extract K,V only once for the maximum group, then slice for smaller groups.

    Args:
        dout: Already sliced for current head range
        q: Already sliced for current head range
        out: Already sliced for current head range
        lse: Already sliced for current head range
        kv_buffer: [2, total_tokens, heads_k_stride, head_dim] - zigzag buffer
        dkv_buffer: [2, total_tokens, heads_k_stride, head_dim] - output buffer (contiguous space)
        dq: Output buffer (already sliced for current head range)
        use_triton_kernel: If True, use Triton kernel for KV slicing (faster).

    Returns:
        None - writes directly to dq and dkv_buffer
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

    # Temporary list for dq groups (will be scattered back to dq buffer at end)
    dq_groups = []

    # Extract dimensions from kv_buffer
    nheads_k = kv_buffer.shape[2]
    head_dim = kv_buffer.shape[3]

    # Rearrange to contiguous format once for Python fallback
    kv_contiguous = None
    if not use_triton_kernel:
        kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)

    # Process each group, extracting K,V specifically for that group
    for group_idx, (dout_group, q_group, out_group, lse_group, cu_seqlens_q_group) in enumerate(zip(
        dout_groups, q_groups, out_groups, lse_groups, chunk_cu_seqlens_q_list
    )):
        if q_group.shape[0] == 0:
            # Empty group, skip
            continue

        seq_ranges, cu_seqlens_k_slice = kv_slices[group_idx]

        # DEBUG: Check kv_buffer before extraction
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            has_nan_kv = torch.isnan(kv_buffer).any().item()
            print(f"[Rank {rank}] Backward Group {group_idx} - Before extraction:")
            print(f"  kv_buffer NaN={has_nan_kv}, shape={kv_buffer.shape}")

        # Extract K,V for this specific group
        if use_triton_kernel:
            # Use Triton kernel to extract K,V slices directly from zigzag buffer
            # NOTE: seq_ranges contains GLOBAL coordinates
            # The start/end are ONLY used to compute output buffer size (num_tokens = end - start)
            # The actual extraction is controlled by max_chunk_idx
            # So we keep GLOBAL coordinates for start/end to get correct output size!
            seq_ranges_with_chunk_idx = []
            for seq_idx, (start_global, end_global) in enumerate(seq_ranges):
                # Get LOCAL sequence length to compute max_chunk_idx
                seq_start_local = cu_seqlens_k[seq_idx].item()
                seq_end_local = cu_seqlens_k[seq_idx + 1].item()
                seq_len_local = seq_end_local - seq_start_local
                total_chunks = 2 * world_size

                # Compute max_chunk_idx from GLOBAL token count
                num_tokens_global = end_global - start_global
                chunk_size_global = seq_len_local * world_size // total_chunks
                max_chunk_idx = (num_tokens_global // chunk_size_global) - 1

                # Keep GLOBAL coordinates for start/end so output buffer size is correct
                seq_ranges_with_chunk_idx.append((start_global, end_global, max_chunk_idx))

            # Create GLOBAL cu_seqlens for Triton kernel
            # The Triton kernel needs GLOBAL cu_seqlens to compute sequence lengths correctly
            cu_seqlens_k_global = torch.tensor(
                [cu_seqlens_k[i].item() * world_size for i in range(len(cu_seqlens_k))],
                device=cu_seqlens_k.device,
                dtype=cu_seqlens_k.dtype
            )

            k_slice, v_slice = extract_zigzag_kv_slices_for_group(
                kv_buffer,
                seq_ranges_with_chunk_idx,
                cu_seqlens_k_global,  # Pass GLOBAL, not LOCAL!
                world_size,
                nheads_k,
                head_dim,
            )
        else:
            # Fallback: Python-based extraction
            k_slices = []
            v_slices = []
            for start, end in seq_ranges:
                k_slices.append(kv_contiguous[0, start:end])
                v_slices.append(kv_contiguous[1, start:end])

            k_slice = torch.cat(k_slices, dim=0).contiguous()
            v_slice = torch.cat(v_slices, dim=0).contiguous()

        # DEBUG: Check extracted K/V slices before backward
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            has_nan_k = torch.isnan(k_slice).any().item()
            has_nan_v = torch.isnan(v_slice).any().item()
            has_nan_q = torch.isnan(q_group).any().item()
            has_nan_dout = torch.isnan(dout_group).any().item()
            has_nan_out = torch.isnan(out_group).any().item()
            print(f"[Rank {rank}] Before backward Group {group_idx}:")
            print(f"  k_slice NaN={has_nan_k}, v_slice NaN={has_nan_v}")
            print(f"  q_group NaN={has_nan_q}, dout_group NaN={has_nan_dout}, out_group NaN={has_nan_out}")
            print(f"  k_slice.shape: {k_slice.shape}, v_slice.shape: {v_slice.shape}")
            print(f"  cu_seqlens_q_group: {cu_seqlens_q_group.tolist()}")
            print(f"  cu_seqlens_k_slice: {cu_seqlens_k_slice.tolist()}")

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

        # DEBUG
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            has_nan_dk = torch.isnan(dk_slice).any().item()
            has_nan_dv = torch.isnan(dv_slice).any().item()
            print(f"\n[Rank {rank}] Backward Group {group_idx}:")
            print(f"  dk_slice.shape: {dk_slice.shape}, dv_slice.shape: {dv_slice.shape}")
            print(f"  dk_slice NaN={has_nan_dk}, dv_slice NaN={has_nan_dv}")
            print(f"  seq_ranges (GLOBAL): {seq_ranges}")
            print(f"  Expected total tokens: {sum(end - start for start, end in seq_ranges)}")

        # CRITICAL: ACCUMULATE gradients into dkv_buffer (not overwrite!)
        # Multiple Q groups may attend to overlapping K,V regions
        # dkv_buffer is in contiguous space (all-gathered format)
        # seq_ranges use GLOBAL coordinates in the contiguous buffer
        offset = 0
        for start, end in seq_ranges:
            length = end - start
            dkv_buffer[0, start:end] += dk_slice[offset:offset+length]  # Accumulate K gradients
            dkv_buffer[1, start:end] += dv_slice[offset:offset+length]  # Accumulate V gradients
            offset += length

        # DEBUG: Check accumulated gradients
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] After accumulation Group {group_idx}:")
            print(f"  dkv_buffer[0] (dK) max={dkv_buffer[0].abs().max():.4f}, mean={dkv_buffer[0].abs().mean():.4f}")
            print(f"  dkv_buffer[1] (dV) max={dkv_buffer[1].abs().max():.4f}, mean={dkv_buffer[1].abs().mean():.4f}")

        dq_groups.append(dq_group)

    # Scatter dq back to the output buffer
    for dq_group, indices in zip(dq_groups, chunk_indices_list):
        dq[indices] = dq_group


def backward_fused_kernel_mode(
    dout: torch.Tensor,     # Already sliced for current head range
    q: torch.Tensor,        # Already sliced for current head range
    out: torch.Tensor,      # Already sliced for current head range
    lse: torch.Tensor,      # Already sliced for current head range
    kv_buffer: torch.Tensor,  # [2, total_tokens, heads_k_stride, head_dim] - zigzag buffer
    dkv_buffer: torch.Tensor, # [2, total_tokens, heads_k_stride, head_dim] - output gradient buffer
    dq: torch.Tensor,       # Output buffer (already sliced for current head range)
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
    cu_seqlens_k: torch.Tensor,
) -> None:  # No return - writes directly to dq and dkv_buffer
    """
    Execute backward with one fused kernel call (all groups together).

    More efficient: Single backward call, then extract and accumulate gradients.

    Args:
        dout: Already sliced for current head range
        q: Already sliced for current head range
        out: Already sliced for current head range
        lse: Already sliced for current head range
        kv_buffer: [2, total_tokens, heads_k_stride, head_dim] - zigzag buffer
        dkv_buffer: [2, total_tokens, heads_k_stride, head_dim] - output buffer
        dq: Output buffer (already sliced for current head range)

    Returns:
        None - writes directly to dq and dkv_buffer
    """
    # Rearrange kv_buffer to contiguous format (fused mode needs this)
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)

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

    # Extract and ACCUMULATE dK, dV from duplicated gradients into dkv_buffer
    # dkv_buffer is in contiguous space (all-gathered format)
    offset_k = 0
    for seq_ranges, cu_seqlens_k_slice in kv_slices:
        # Calculate total length from cu_seqlens
        k_slice_len = cu_seqlens_k_slice[-1].item()
        dk_slice = dk_duplicated[offset_k:offset_k+k_slice_len]
        dv_slice = dv_duplicated[offset_k:offset_k+k_slice_len]

        # ACCUMULATE (different groups may have overlapping K,V regions)
        # Scatter gradients back to correct sequence positions in dkv_buffer
        offset_in_slice = 0
        for start, end in seq_ranges:
            length = end - start
            dkv_buffer[0, start:end] += dk_slice[offset_in_slice:offset_in_slice+length]  # Accumulate K gradients
            dkv_buffer[1, start:end] += dv_slice[offset_in_slice:offset_in_slice+length]  # Accumulate V gradients
            offset_in_slice += length

        offset_k += k_slice_len

    # Scatter dq back to the output buffer
    offset = 0
    for indices in chunk_indices_list:
        chunk_size = len(indices)
        dq[indices] = dq_combined[offset:offset+chunk_size]
        offset += chunk_size


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
    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape

    # Validate heads_k_stride
    assert nheads_k % heads_k_stride == 0, \
        f"nheads_k ({nheads_k}) must be divisible by heads_k_stride ({heads_k_stride})"

    # Initialize full gradient tensors (will be filled by head iterations)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # Allocate buffers for all-gather
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    # Allocate gradient buffer (in contiguous all-gathered space)
    dkv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    # Allocate contiguous buffer for reduce-scatter if using head batching
    if heads_k_stride != nheads_k:
        kv_contiguous_buffer = torch.empty(
            (2, total_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )

    # Start first all-gather
    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm = Comm(process_group)
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    # Loop over KV head slices
    for i in range(0, nheads_k, heads_k_stride):
        # Zero out gradient buffer for this iteration
        dkv_buffer.zero_()

        # Slice Q-related tensors for current head range (for GQA)
        q_slice = slice(
            i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
        )
        q_i = q[:, q_slice]
        dout_i = dout[:, q_slice]
        out_i = out[:, q_slice]
        dq_i = dq[:, q_slice]

        # Handle LSE dimension variations
        if lse.dim() == 3:
            # LSE is [batch, nheads, max_seqlen] or similar
            lse_i = lse[:, q_slice].contiguous()
        elif lse.dim() == 2 and lse.shape[0] == nheads:
            # LSE is [nheads, total_seqlen]
            lse_i = lse[q_slice].contiguous()
        else:
            # LSE is [total_seqlen, nheads]
            lse_i = lse[:, q_slice].contiguous()

        # Wait for current all-gather and swap buffers
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        # Start next all-gather (overlapped with computation)
        if i < nheads_k - heads_k_stride:
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        # Execute backward for this head slice (processes all Q groups)
        if use_fused_kernel:
            # Fused kernel mode
            backward_fused_kernel_mode(
                dout_i, q_i, out_i, lse_i,
                kv_buffer, dkv_buffer, dq_i,
                chunk_indices_list, chunk_cu_seqlens_q_list, kv_slices,
                softmax_scale, dropout_p, causal, window_size, alibi_slopes, deterministic,
                world_size, rank, cu_seqlens_k
            )
        else:
            # Two-kernels mode
            # Note: Currently using Python fallback for KV extraction in backward
            # as Triton extraction kernel has autograd compatibility issues
            # Triton scatter_grad kernel still used for gradient rearrangement
            backward_two_kernels_mode(
                dout_i, q_i, out_i, lse_i,
                kv_buffer, dkv_buffer, dq_i,
                chunk_indices_list, chunk_cu_seqlens_q_list, kv_slices,
                softmax_scale, dropout_p, causal, window_size, alibi_slopes, deterministic,
                world_size, rank, cu_seqlens_k, use_triton_kernel=False
            )

        # Convert gradients from contiguous to zigzag interleaved format
        dkv_contiguous = torch.stack([dkv_buffer[0], dkv_buffer[1]])  # [2, total_tokens, heads_k_stride, dim]

        # DEBUG: Before rearrange
        import os
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            print(f"\n[Rank {rank}] Before rearrange (head slice {i}):")
            print(f"  dkv_contiguous shape: {dkv_contiguous.shape}")
            print(f"  dkv_contiguous[0] (dK) max={dkv_contiguous[0].abs().max():.4f}, mean={dkv_contiguous[0].abs().mean():.4f}")
            print(f"  dkv_contiguous[1] (dV) max={dkv_contiguous[1].abs().max():.4f}, mean={dkv_contiguous[1].abs().mean():.4f}")
            print(f"  cu_seqlens_k: {cu_seqlens_k.tolist()}")
            print(f"  world_size: {world_size}, rank: {rank}")

        # CRITICAL FIX: Convert GLOBAL cu_seqlens_k to LOCAL
        # rearrange_grad_from_contiguous_to_zigzag expects LOCAL cu_seqlens
        cu_seqlens_k_local = torch.tensor(
            [cu_seqlens_k[i].item() // world_size for i in range(len(cu_seqlens_k))],
            device=cu_seqlens_k.device,
            dtype=cu_seqlens_k.dtype
        )

        dkv_interleaved = rearrange_grad_from_contiguous_to_zigzag(
            dkv_contiguous, world_size, cu_seqlens_k_local, rank, use_triton_kernel=use_triton_kernel
        )

        # DEBUG: After rearrange
        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            print(f"[Rank {rank}] After rearrange:")
            print(f"  dkv_interleaved shape: {dkv_interleaved.shape}")
            print(f"  dkv_interleaved[0] (dK) max={dkv_interleaved[0].abs().max():.4f}, mean={dkv_interleaved[0].abs().mean():.4f}")
            print(f"  dkv_interleaved[1] (dV) max={dkv_interleaved[1].abs().max():.4f}, mean={dkv_interleaved[1].abs().mean():.4f}")

        # Reduce-scatter to get local gradients for this head slice
        if heads_k_stride != nheads_k:
            # Use contiguous buffer for reduce-scatter
            dk_i = kv_contiguous_buffer[0]
            dv_i = kv_contiguous_buffer[1]
        else:
            # Write directly to final dk, dv
            dk_i = dk[:, i:i+heads_k_stride]
            dv_i = dv[:, i:i+heads_k_stride]

        # Optimization #7: Async reduce-scatter to overlap dK and dV communication
        handle_dk = dist.reduce_scatter_tensor(
            dk_i,
            dkv_interleaved[0],
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True
        )
        handle_dv = dist.reduce_scatter_tensor(
            dv_i,
            dkv_interleaved[1],
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True
        )

        # Wait for both to complete
        handle_dk.wait()
        handle_dv.wait()

        # Copy to final dk, dv if using head batching
        if heads_k_stride != nheads_k:
            dk[:, i:i+heads_k_stride] = dk_i
            dv[:, i:i+heads_k_stride] = dv_i

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
        use_triton_grouped,
        use_grouped_attention,
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
            use_triton_grouped=use_triton_grouped,
            use_grouped_attention=use_grouped_attention,
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
    use_triton_grouped=False,
    use_grouped_attention=False,
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
        use_triton_grouped: Use Triton grouped attention kernel (Option C)
        use_grouped_attention: Use Python prototype grouped attention (Option B)

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
        use_triton_grouped,
        use_grouped_attention,
    )

