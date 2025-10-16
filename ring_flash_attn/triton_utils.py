import torch
import triton
import triton.language as tl

# Suppress torch.compile warnings for .item() calls in wrapper functions
# These functions are not compiled, but imported modules might trigger dynamo
import warnings
warnings.filterwarnings('ignore', message='.*Graph break from.*Tensor.item.*')


@triton.jit
def flatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_batch,
    stride_lse_nheads,
    stride_lse_seqlen,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_batch * stride_lse_batch + pid_head * stride_lse_nheads
    OUT = OUT + pid_head * stride_out_nheads + start_idx * stride_out_seqlen

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def flatten_varlen_lse(lse, cu_seqlens):
    """
    Arguments:
        lse: (batch_size, nheads, max_seqlen)
        cu_seqlens: (batch_size + 1,)
    Return:
        flatten_lse: (nheads, total_seqlen)
    """
    total_seqlen = cu_seqlens[-1]
    batch_size, nheads, max_seqlen = lse.shape
    output = torch.empty((nheads, total_seqlen), dtype=lse.dtype, device=lse.device)

    grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads)
    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        flatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            BLOCK_M,
        )
    return output


@triton.jit
def unflatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_batch,
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_seqlen,
    stride_lse_nheads,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_head * stride_lse_nheads + start_idx * stride_lse_seqlen
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    """
    Arguments:
        lse: (total_seqlen, nheads, 1)
        cu_seqlens: (batch_size + 1,)
        max_seqlen: int
    Return:
        unflatten_lse: (batch_size, nheads, max_seqlen)
    """
    lse = lse.unsqueeze(dim=-1)
    batch_size = len(cu_seqlens) - 1
    nheads = lse.shape[1]
    output = torch.empty(
        (batch_size, nheads, max_seqlen),
        dtype=lse.dtype,
        device=lse.device,
    )

    grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads)
    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        unflatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            BLOCK_M,
        )
    return output


@triton.jit
def extract_zigzag_slice_kernel_v2(
    # Input/Output pointers
    KV_BUFFER,  # [total_tokens, nheads_k, head_dim] - zigzag interleaved format (K or V only)
    KV_OUT,     # [total_output_tokens, nheads_k, head_dim] - contiguous output
    CU_SEQLENS_GLOBAL,  # [num_seqs + 1] - global sequence boundaries
    # Slice parameters
    max_chunk_idx,      # Maximum chunk index to extract (for causal attention)
    world_size,
    local_tokens_per_rank,  # tokens per rank in zigzag format
    # Sequence info
    seq_idx,
    seq_offset_in_zigzag,  # Offset of this sequence within the zigzag buffer
    # Strides
    stride_kv_tokens,
    stride_kv_heads,
    stride_kv_dim,
    stride_out_tokens,
    stride_out_heads,
    stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Extract K or V slice from zigzag interleaved buffer (FIXED VERSION).

    This version takes separate K and V buffers instead of a combined [2, ...] tensor,
    eliminating the large offset issue that caused NaN in V extraction.

    Zigzag format: Each rank r contributes [chunk_r, chunk_(2*world_size-1-r)]
    This kernel extracts chunks [0, 1, ..., max_chunk_idx] in contiguous order for one sequence.
    """
    pid = tl.program_id(axis=0)  # Token block
    head_idx = tl.program_id(axis=1)  # Head index

    # Load sequence boundaries
    seq_start_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx)
    seq_end_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx + 1)
    seq_len = seq_end_global - seq_start_global

    total_chunks = 2 * world_size
    chunk_size = seq_len // total_chunks

    # Calculate which token we're processing (within this sequence's output)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Only process tokens within the desired range [0, (max_chunk_idx+1)*chunk_size)
    num_tokens_needed = (max_chunk_idx + 1) * chunk_size
    mask = token_idx < num_tokens_needed

    # For each output token, find its source location in zigzag buffer
    # Output token belongs to chunk: token_idx // chunk_size
    chunk_idx = token_idx // chunk_size
    offset_in_chunk = token_idx % chunk_size

    # Determine which rank owns this chunk in zigzag format
    rank_owner = tl.where(
        chunk_idx < world_size,
        chunk_idx,
        2 * world_size - 1 - chunk_idx
    )

    # Within that rank's contribution, is this the first or second chunk?
    is_first_chunk = chunk_idx < world_size

    # Calculate position within the rank's local buffer
    chunk_offset_in_seq = tl.where(is_first_chunk, 0, chunk_size)

    source_token_idx = (
        rank_owner * local_tokens_per_rank +
        seq_offset_in_zigzag +
        chunk_offset_in_seq +
        offset_in_chunk
    )

    # Load from buffer and store to output
    dim_idx = tl.arange(0, HEAD_DIM)

    # Source address (NO kv_type offset - much simpler!)
    kv_offset = (
        source_token_idx[:, None] * stride_kv_tokens +
        head_idx * stride_kv_heads +
        dim_idx[None, :] * stride_kv_dim
    )

    # Output address
    out_offset = (
        token_idx[:, None] * stride_out_tokens +
        head_idx * stride_out_heads +
        dim_idx[None, :] * stride_out_dim
    )

    # Load and store
    kv_data = tl.load(KV_BUFFER + kv_offset, mask=mask[:, None], other=0.0)
    tl.store(KV_OUT + out_offset, kv_data, mask=mask[:, None])


@triton.jit
def extract_zigzag_kv_slice_kernel(
    # Input/Output pointers
    KV_ZIGZAG,  # [2, total_tokens, nheads_k, head_dim] - zigzag interleaved format
    KV_OUT,     # [total_output_tokens, nheads_k, head_dim] - contiguous output
    CU_SEQLENS_GLOBAL,  # [num_seqs + 1] - global sequence boundaries
    # Slice parameters
    max_chunk_idx,      # Maximum chunk index to extract (for causal attention)
    world_size,
    local_tokens_per_rank,  # tokens per rank in zigzag format
    kv_type,  # 0 for K, 1 for V
    # Sequence info
    seq_idx,
    seq_offset_in_zigzag,  # ADDED: Offset of this sequence within the zigzag buffer
    # Strides
    stride_kv_kv,  # ADDED: Stride for K/V selector dimension (dim 0)
    stride_kv_tokens,
    stride_kv_heads,
    stride_kv_dim,
    stride_out_tokens,
    stride_out_heads,
    stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Extract K or V slice from zigzag interleaved all-gathered buffer.

    Zigzag format: Each rank r contributes [chunk_r, chunk_(2*world_size-1-r)]
    For multiple sequences, within each rank: [seq0_chunk_r, seq0_chunk_(2*ws-1-r), seq1_chunk_r, seq1_chunk_(2*ws-1-r), ...]

    After all-gather: [r0_all_seqs, r1_all_seqs, r2_all_seqs, ...]

    This kernel extracts chunks [0, 1, ..., max_chunk_idx] in contiguous order for one sequence.
    """
    pid = tl.program_id(axis=0)  # Token block
    head_idx = tl.program_id(axis=1)  # Head index

    # Load sequence boundaries
    seq_start_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx)
    seq_end_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx + 1)
    seq_len = seq_end_global - seq_start_global

    total_chunks = 2 * world_size
    chunk_size = seq_len // total_chunks

    # Calculate which token we're processing (within this sequence's output)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Only process tokens within the desired range [0, (max_chunk_idx+1)*chunk_size)
    num_tokens_needed = (max_chunk_idx + 1) * chunk_size
    mask = token_idx < num_tokens_needed

    # For each output token, find its source location in zigzag buffer
    # Output token belongs to chunk: token_idx // chunk_size
    chunk_idx = token_idx // chunk_size
    offset_in_chunk = token_idx % chunk_size

    # Determine which rank owns this chunk in zigzag format
    # chunk_idx < world_size: owned by rank = chunk_idx (first chunk from rank)
    # chunk_idx >= world_size: owned by rank = 2*world_size - 1 - chunk_idx (second chunk from rank)
    rank_owner = tl.where(
        chunk_idx < world_size,
        chunk_idx,
        2 * world_size - 1 - chunk_idx
    )

    # Within that rank's contribution, is this the first or second chunk (for THIS sequence)?
    is_first_chunk = chunk_idx < world_size

    # Calculate position within the rank's local buffer
    # Each rank's contribution starts at: rank * local_tokens_per_rank
    # Within the rank, this sequence starts at: seq_offset_in_zigzag (cumulative tokens from previous seqs)
    # Within the sequence in this rank: first chunk at 0, second chunk at chunk_size
    chunk_offset_in_seq = tl.where(is_first_chunk, 0, chunk_size)

    source_token_idx = (
        rank_owner * local_tokens_per_rank +  # Start of this rank's contribution
        seq_offset_in_zigzag +                 # Offset to this sequence within the rank
        chunk_offset_in_seq +                  # Offset to the chunk within the sequence
        offset_in_chunk                        # Offset within the chunk
    )

    # Load from zigzag buffer and store to output
    # Load all dimensions for this head
    dim_idx = tl.arange(0, HEAD_DIM)

    # Source address in zigzag buffer
    kv_offset = (
        kv_type * stride_kv_kv +  # K or V offset (dimension 0)
        source_token_idx[:, None] * stride_kv_tokens +
        head_idx * stride_kv_heads +
        dim_idx[None, :] * stride_kv_dim
    )

    # Output address
    out_offset = (
        token_idx[:, None] * stride_out_tokens +
        head_idx * stride_out_heads +
        dim_idx[None, :] * stride_out_dim
    )

    # Load and store
    kv_data = tl.load(KV_ZIGZAG + kv_offset, mask=mask[:, None], other=0.0)
    tl.store(KV_OUT + out_offset, kv_data, mask=mask[:, None])


def extract_zigzag_kv_slices_for_group(
    kv_buffer: torch.Tensor,
    seq_ranges: list,
    cu_seqlens_global: torch.Tensor,
    world_size: int,
    nheads_k: int,
    head_dim: int,
    use_v2_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract K,V slices from zigzag interleaved buffer using Triton kernel.

    Args:
        kv_buffer: [2, total_tokens, nheads_k, head_dim] - all-gathered zigzag buffer
        seq_ranges: List of (start, end, max_chunk_idx) tuples for each sequence
        cu_seqlens_global: [num_seqs + 1] - global cumulative sequence lengths
        world_size: Number of ranks
        nheads_k: Number of KV heads
        head_dim: Head dimension
        use_v2_kernel: If True, use fixed v2 kernel (splits K/V). If False, use original (for comparison).

    Returns:
        k_slice: Extracted K tensor [total_output_tokens, nheads_k, head_dim]
        v_slice: Extracted V tensor [total_output_tokens, nheads_k, head_dim]
    """
    # Calculate total output size
    total_output_tokens = sum(end - start for start, end, _ in seq_ranges)

    # Allocate output buffers
    k_out = torch.empty(
        (total_output_tokens, nheads_k, head_dim),
        dtype=kv_buffer.dtype,
        device=kv_buffer.device
    )
    v_out = torch.empty(
        (total_output_tokens, nheads_k, head_dim),
        dtype=kv_buffer.dtype,
        device=kv_buffer.device
    )

    total_tokens = kv_buffer.shape[1]
    local_tokens_per_rank = total_tokens // world_size
    total_chunks = 2 * world_size

    # Calculate sequence offsets within each rank's zigzag buffer
    # Each rank has all sequences interleaved: [seq0_chunks, seq1_chunks, ...]
    # We need to know the cumulative offset for each sequence
    seq_offsets_in_zigzag = [0]
    num_seqs = len(cu_seqlens_global) - 1
    for i in range(num_seqs):
        seq_len = (cu_seqlens_global[i + 1] - cu_seqlens_global[i]).item()
        tokens_per_rank_for_seq = 2 * (seq_len // total_chunks)  # Each rank has 2 chunks
        seq_offsets_in_zigzag.append(seq_offsets_in_zigzag[-1] + tokens_per_rank_for_seq)

    BLOCK_SIZE = 128

    # DEBUG
    import os
    if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
        k_has_nan = torch.isnan(kv_buffer[0]).any().item()
        v_has_nan = torch.isnan(kv_buffer[1]).any().item()
        print(f"[extract_zigzag_kv_slices_for_group] Using {'v2' if use_v2_kernel else 'v1'} kernel")
        print(f"[extract_zigzag_kv_slices_for_group] kv_buffer.shape: {kv_buffer.shape}")
        print(f"[extract_zigzag_kv_slices_for_group] kv_buffer strides: {kv_buffer.stride()}")
        print(f"[extract_zigzag_kv_slices_for_group] kv_buffer is_contiguous: {kv_buffer.is_contiguous()}")
        print(f"[extract_zigzag_kv_slices_for_group] kv_buffer K NaN={k_has_nan}, V NaN={v_has_nan}")
        if not use_v2_kernel:
            print(f"[extract_zigzag_kv_slices_for_group] Expected K offset: 0, V offset: {kv_buffer.stride(0)}")
            print(f"[extract_zigzag_kv_slices_for_group] Buffer size: {kv_buffer.numel()} elements, {kv_buffer.element_size() * kv_buffer.numel()} bytes")
            print(f"[extract_zigzag_kv_slices_for_group] Max V offset would be: {kv_buffer.stride(0) + (kv_buffer.shape[1]-1) * kv_buffer.stride(1)}")

        # Test direct indexing of V buffer
        print(f"[extract_zigzag_kv_slices_for_group] Testing direct V buffer access:")
        v_buffer = kv_buffer[1]  # [total_tokens, nheads_k, head_dim]
        print(f"  v_buffer shape: {v_buffer.shape}, strides: {v_buffer.stride()}")
        print(f"  v_buffer is_contiguous: {v_buffer.is_contiguous()}")
        print(f"  v_buffer has NaN: {torch.isnan(v_buffer).any().item()}")
        # Try to access a small slice
        if v_buffer.shape[0] > 0:
            v_slice_test = v_buffer[:min(10, v_buffer.shape[0])]
            print(f"  v_buffer[:10] has NaN: {torch.isnan(v_slice_test).any().item()}")

    # FIX 1: Split KV buffer for v2 kernel
    if use_v2_kernel:
        # Extract separate K and V buffers (contiguous)
        k_buffer = kv_buffer[0].contiguous()  # [total_tokens, nheads_k, head_dim]
        v_buffer = kv_buffer[1].contiguous()  # [total_tokens, nheads_k, head_dim]

        if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
            print(f"[extract_zigzag_kv_slices_for_group] Split buffers:")
            print(f"  k_buffer.shape: {k_buffer.shape}, is_contiguous: {k_buffer.is_contiguous()}")
            print(f"  v_buffer.shape: {v_buffer.shape}, is_contiguous: {v_buffer.is_contiguous()}")
            print(f"  k_buffer has NaN: {torch.isnan(k_buffer).any().item()}")
            print(f"  v_buffer has NaN: {torch.isnan(v_buffer).any().item()}")

    # Process each sequence
    output_offset = 0
    for seq_idx, (start, end, max_chunk_idx) in enumerate(seq_ranges):
        num_tokens = end - start
        seq_offset_in_zigzag = seq_offsets_in_zigzag[seq_idx]

        # Grid: (num_token_blocks, nheads_k)
        grid = lambda META: (
            triton.cdiv(num_tokens, BLOCK_SIZE),
            nheads_k,
        )

        with torch.cuda.device(kv_buffer.device.index):
            if use_v2_kernel:
                # V2 KERNEL: Use split K/V buffers (FIX 1)
                # Extract K for this sequence
                extract_zigzag_slice_kernel_v2[grid](
                    k_buffer,  # Separate K buffer (no offset needed)
                    k_out[output_offset:output_offset + num_tokens],
                    cu_seqlens_global,
                    max_chunk_idx,
                    world_size,
                    local_tokens_per_rank,
                    seq_idx,
                    seq_offset_in_zigzag,
                    # K buffer strides (simpler - no K/V dimension)
                    k_buffer.stride(0),  # tokens dimension
                    k_buffer.stride(1),  # heads dimension
                    k_buffer.stride(2),  # dim dimension
                    # Output strides
                    k_out.stride(0),
                    k_out.stride(1),
                    k_out.stride(2),
                    BLOCK_SIZE=BLOCK_SIZE,
                    HEAD_DIM=head_dim,
                )

                # Extract V for this sequence
                extract_zigzag_slice_kernel_v2[grid](
                    v_buffer,  # Separate V buffer (no offset needed)
                    v_out[output_offset:output_offset + num_tokens],
                    cu_seqlens_global,
                    max_chunk_idx,
                    world_size,
                    local_tokens_per_rank,
                    seq_idx,
                    seq_offset_in_zigzag,
                    # V buffer strides (identical to K)
                    v_buffer.stride(0),  # tokens dimension
                    v_buffer.stride(1),  # heads dimension
                    v_buffer.stride(2),  # dim dimension
                    # Output strides
                    v_out.stride(0),
                    v_out.stride(1),
                    v_out.stride(2),
                    BLOCK_SIZE=BLOCK_SIZE,
                    HEAD_DIM=head_dim,
                )
            else:
                # V1 KERNEL: Original (may have V extraction bug)
                # Extract K for this sequence
                extract_zigzag_kv_slice_kernel[grid](
                    kv_buffer,
                    k_out[output_offset:output_offset + num_tokens],
                    cu_seqlens_global,
                    max_chunk_idx,
                    world_size,
                    local_tokens_per_rank,
                    0,  # K
                    seq_idx,
                    seq_offset_in_zigzag,
                    # KV buffer strides
                    kv_buffer.stride(0),  # K/V selector dimension
                    kv_buffer.stride(1),  # tokens dimension
                    kv_buffer.stride(2),  # heads dimension
                    kv_buffer.stride(3),  # dim dimension
                    # Output strides
                    k_out.stride(0),
                    k_out.stride(1),
                    k_out.stride(2),
                    BLOCK_SIZE=BLOCK_SIZE,
                    HEAD_DIM=head_dim,
                )

                # Extract V for this sequence
                extract_zigzag_kv_slice_kernel[grid](
                    kv_buffer,
                    v_out[output_offset:output_offset + num_tokens],
                    cu_seqlens_global,
                    max_chunk_idx,
                    world_size,
                    local_tokens_per_rank,
                    1,  # V
                    seq_idx,
                    seq_offset_in_zigzag,
                    # KV buffer strides
                    kv_buffer.stride(0),  # K/V selector dimension
                    kv_buffer.stride(1),  # tokens dimension
                    kv_buffer.stride(2),  # heads dimension
                    kv_buffer.stride(3),  # dim dimension
                    # Output strides
                    v_out.stride(0),
                    v_out.stride(1),
                    v_out.stride(2),
                    BLOCK_SIZE=BLOCK_SIZE,
                    HEAD_DIM=head_dim,
                )

            # DEBUG: Check outputs immediately after extraction
            if os.environ.get('DEBUG_ZIGZAG', '0') == '1':
                k_slice = k_out[output_offset:output_offset + num_tokens]
                v_slice = v_out[output_offset:output_offset + num_tokens]
                k_nan = torch.isnan(k_slice).any().item()
                v_nan = torch.isnan(v_slice).any().item()
                print(f"[extract_zigzag_kv_slices_for_group] seq_idx={seq_idx}, max_chunk_idx={max_chunk_idx}")
                print(f"  num_tokens={num_tokens}, seq_offset_in_zigzag={seq_offset_in_zigzag}")
                print(f"  K slice NaN={k_nan}, V slice NaN={v_nan}")
                print(f"  Strides passed to kernel:")
                print(f"    stride_kv_kv={kv_buffer.stride(0)}")
                print(f"    stride_kv_tokens={kv_buffer.stride(1)}")
                print(f"    stride_kv_heads={kv_buffer.stride(2)}")
                print(f"    stride_kv_dim={kv_buffer.stride(3)}")
                if v_nan:
                    print(f"  V slice shape: {v_slice.shape}")
                    print(f"  V buffer source shape: {kv_buffer[1].shape}")
                    print(f"  V nan count: {torch.isnan(v_slice).sum().item()} / {v_slice.numel()}")
                    # Compare with Python-based extraction
                    print(f"  Comparing with Python reference extraction...")
                    v_ref = kv_buffer[1]  # [total_tokens, nheads_k, head_dim]
                    # Manually extract using the same logic as the kernel
                    total_chunks = 2 * world_size
                    seq_len = (cu_seqlens_global[seq_idx + 1] - cu_seqlens_global[seq_idx]).item()
                    chunk_size = seq_len // total_chunks
                    num_tokens_needed = (max_chunk_idx + 1) * chunk_size
                    print(f"  seq_len={seq_len}, chunk_size={chunk_size}, num_tokens_needed={num_tokens_needed}")
                    # Check first few tokens manually
                    for token_idx in range(min(5, num_tokens)):
                        chunk_idx = token_idx // chunk_size
                        offset_in_chunk = token_idx % chunk_size
                        rank_owner = chunk_idx if chunk_idx < world_size else 2 * world_size - 1 - chunk_idx
                        is_first_chunk = chunk_idx < world_size
                        chunk_offset_in_seq = 0 if is_first_chunk else chunk_size
                        source_token_idx = (rank_owner * local_tokens_per_rank +
                                          seq_offset_in_zigzag +
                                          chunk_offset_in_seq +
                                          offset_in_chunk)
                        v_ref_val = v_ref[source_token_idx, 0, 0]  # First head, first dim
                        v_out_val = v_slice[token_idx, 0, 0]
                        print(f"    token {token_idx}: source_idx={source_token_idx}, v_ref={v_ref_val:.6f}, v_out={v_out_val:.6f}")

        output_offset += num_tokens

    return k_out, v_out


@triton.jit
def scatter_grad_to_zigzag_kernel(
    # Input/Output pointers
    GRAD_CONTIGUOUS,  # [total_tokens, nheads_k, head_dim] - contiguous gradient
    GRAD_ZIGZAG,      # [local_tokens, nheads_k, head_dim] - zigzag output (for this rank)
    CU_SEQLENS_GLOBAL,  # [num_seqs + 1]
    # Parameters
    rank,
    world_size,
    seq_idx,  # Which sequence we're processing
    seq_offset_in_zigzag,  # Offset of this sequence in the local zigzag buffer
    # Strides
    stride_grad_cont_tokens,
    stride_grad_cont_heads,
    stride_grad_cont_dim,
    stride_grad_zig_tokens,
    stride_grad_zig_heads,
    stride_grad_zig_dim,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Optimization #5: Fused gradient scatter kernel

    Scatter gradients from contiguous format back to zigzag interleaved format.
    This kernel is the reverse of extract_zigzag_kv_slice_kernel.

    For each rank, we need to extract gradients for:
    - chunk[rank] (early chunk)
    - chunk[2*world_size - 1 - rank] (late chunk)

    And concatenate them in zigzag interleaved format.
    """
    pid_token = tl.program_id(axis=0)  # Token block
    pid_head = tl.program_id(axis=1)   # Head index

    # Load sequence info
    seq_start_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx)
    seq_end_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx + 1)
    seq_len = seq_end_global - seq_start_global

    total_chunks = 2 * world_size
    chunk_size = seq_len // total_chunks

    # This rank owns two chunks
    chunk_idx_0 = rank  # Early chunk
    chunk_idx_1 = 2 * world_size - 1 - rank  # Late chunk

    # Token indices within local buffer (2 chunks concatenated)
    local_tokens_per_seq = 2 * chunk_size
    token_idx_local = pid_token * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx_local < local_tokens_per_seq

    # Determine which chunk this local token belongs to
    is_first_chunk = token_idx_local < chunk_size
    chunk_idx = tl.where(is_first_chunk, chunk_idx_0, chunk_idx_1)
    offset_in_chunk = tl.where(is_first_chunk, token_idx_local, token_idx_local - chunk_size)

    # Source position in contiguous gradient
    source_token_idx = seq_start_global + chunk_idx * chunk_size + offset_in_chunk

    # Dimension indices
    dim_idx = tl.arange(0, HEAD_DIM)

    # Source address in contiguous gradient
    grad_cont_offset = (
        source_token_idx[:, None] * stride_grad_cont_tokens +
        pid_head * stride_grad_cont_heads +
        dim_idx[None, :] * stride_grad_cont_dim
    )

    # Destination address in zigzag gradient (local buffer)
    # seq_offset_in_zigzag is the cumulative offset from previous sequences
    dest_token_idx = seq_offset_in_zigzag + token_idx_local

    grad_zig_offset = (
        dest_token_idx[:, None] * stride_grad_zig_tokens +
        pid_head * stride_grad_zig_heads +
        dim_idx[None, :] * stride_grad_zig_dim
    )

    # Load and store (no atomic needed - each location written once)
    grad_data = tl.load(GRAD_CONTIGUOUS + grad_cont_offset, mask=mask[:, None], other=0.0)
    tl.store(GRAD_ZIGZAG + grad_zig_offset, grad_data, mask=mask[:, None])


def scatter_grad_to_zigzag(
    grad_contiguous: torch.Tensor,  # [total_tokens, nheads_k, head_dim]
    cu_seqlens_global: torch.Tensor,  # [num_seqs + 1]
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Python wrapper for scatter_grad_to_zigzag_kernel.

    Converts gradients from contiguous format back to zigzag interleaved format.

    Args:
        grad_contiguous: Gradients in contiguous format
        cu_seqlens_global: Global sequence boundaries
        rank: Current rank
        world_size: Number of ranks

    Returns:
        grad_zigzag: Gradients in zigzag interleaved format [local_tokens, nheads_k, head_dim]
    """
    total_tokens, nheads_k, head_dim = grad_contiguous.shape
    num_seqs = len(cu_seqlens_global) - 1
    total_chunks = 2 * world_size

    # Calculate local tokens and sequence offsets (each rank gets 2 chunks per sequence)
    local_tokens = 0
    seq_offsets_in_zigzag = [0]
    for i in range(num_seqs):
        seq_len = (cu_seqlens_global[i + 1] - cu_seqlens_global[i]).item()
        chunk_size = seq_len // total_chunks
        tokens_per_rank_for_seq = 2 * chunk_size
        local_tokens += tokens_per_rank_for_seq
        seq_offsets_in_zigzag.append(local_tokens)

    # Allocate output
    grad_zigzag = torch.empty(
        (local_tokens, nheads_k, head_dim),
        dtype=grad_contiguous.dtype,
        device=grad_contiguous.device
    )

    BLOCK_SIZE = 128

    # Launch kernel for each sequence
    for seq_idx in range(num_seqs):
        seq_len = (cu_seqlens_global[seq_idx + 1] - cu_seqlens_global[seq_idx]).item()
        local_tokens_per_seq = 2 * (seq_len // total_chunks)
        seq_offset_in_zigzag = seq_offsets_in_zigzag[seq_idx]

        grid = lambda META: (
            triton.cdiv(local_tokens_per_seq, BLOCK_SIZE),
            nheads_k,
        )

        with torch.cuda.device(grad_contiguous.device.index):
            scatter_grad_to_zigzag_kernel[grid](
                grad_contiguous,
                grad_zigzag,
                cu_seqlens_global,
                rank,
                world_size,
                seq_idx,  # ADDED
                seq_offset_in_zigzag,  # ADDED
                # Contiguous grad strides
                grad_contiguous.stride(0),
                grad_contiguous.stride(1),
                grad_contiguous.stride(2),
                # Zigzag grad strides
                grad_zigzag.stride(0),
                grad_zigzag.stride(1),
                grad_zigzag.stride(2),
                BLOCK_SIZE=BLOCK_SIZE,
                HEAD_DIM=head_dim,
            )

    return grad_zigzag
