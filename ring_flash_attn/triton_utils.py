import torch
import triton
import triton.language as tl


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
    Extract K or V slice from zigzag interleaved all-gathered buffer.

    Zigzag format: Each rank r contributes [chunk_r, chunk_(2*world_size-1-r)]
    After all-gather: [r0_c0, r0_c7, r1_c1, r1_c6, r2_c2, r2_c5, r3_c3, r3_c4]

    This kernel extracts chunks [0, 1, ..., max_chunk_idx] in contiguous order.
    """
    pid = tl.program_id(axis=0)  # Token block
    head_idx = tl.program_id(axis=1)  # Head index

    # Load sequence boundaries
    seq_start_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx)
    seq_end_global = tl.load(CU_SEQLENS_GLOBAL + seq_idx + 1)
    seq_len = seq_end_global - seq_start_global

    total_chunks = 2 * world_size
    chunk_size = seq_len // total_chunks

    # Calculate which token we're processing
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

    # Within that rank's contribution, is this the first or second chunk?
    is_first_chunk = chunk_idx < world_size

    # Calculate position within the rank's local buffer
    # First chunk from rank r: starts at rank * local_tokens_per_rank
    # Second chunk from rank r: starts at rank * local_tokens_per_rank + chunk_size
    chunk_offset_in_rank = tl.where(is_first_chunk, 0, chunk_size)
    source_token_idx = rank_owner * local_tokens_per_rank + chunk_offset_in_rank + offset_in_chunk

    # Add sequence offset (all sequences are concatenated)
    source_token_idx = source_token_idx + seq_start_global

    # Load from zigzag buffer and store to output
    # Load all dimensions for this head
    dim_idx = tl.arange(0, HEAD_DIM)

    # Source address in zigzag buffer
    kv_offset = (
        kv_type * stride_kv_tokens * (local_tokens_per_rank * world_size) +  # K or V offset
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

    BLOCK_SIZE = 128

    # Process each sequence
    output_offset = 0
    for seq_idx, (start, end, max_chunk_idx) in enumerate(seq_ranges):
        num_tokens = end - start

        # Grid: (num_token_blocks, nheads_k)
        grid = lambda META: (
            triton.cdiv(num_tokens, BLOCK_SIZE),
            nheads_k,
        )

        # Extract K for this sequence
        with torch.cuda.device(kv_buffer.device.index):
            extract_zigzag_kv_slice_kernel[grid](
                kv_buffer,
                k_out[output_offset:output_offset + num_tokens],
                cu_seqlens_global,
                max_chunk_idx,
                world_size,
                local_tokens_per_rank,
                0,  # K
                seq_idx,
                # KV buffer strides (first dim is K/V selector)
                kv_buffer.stride(1),
                kv_buffer.stride(2),
                kv_buffer.stride(3),
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
                # KV buffer strides
                kv_buffer.stride(1),
                kv_buffer.stride(2),
                kv_buffer.stride(3),
                # Output strides
                v_out.stride(0),
                v_out.stride(1),
                v_out.stride(2),
                BLOCK_SIZE=BLOCK_SIZE,
                HEAD_DIM=head_dim,
            )

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
    pid_seq = tl.program_id(axis=2)    # Sequence index

    # Load sequence info
    seq_start_global = tl.load(CU_SEQLENS_GLOBAL + pid_seq)
    seq_end_global = tl.load(CU_SEQLENS_GLOBAL + pid_seq + 1)
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
    # Need to account for sequence offset in local buffer
    # (previous sequences' tokens come first)
    local_seq_offset = 0
    for i in range(pid_seq):
        prev_seq_start = tl.load(CU_SEQLENS_GLOBAL + i)
        prev_seq_end = tl.load(CU_SEQLENS_GLOBAL + i + 1)
        prev_seq_len = prev_seq_end - prev_seq_start
        local_seq_offset += 2 * (prev_seq_len // total_chunks)

    dest_token_idx = local_seq_offset + token_idx_local

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

    # Calculate local tokens (each rank gets 2 chunks per sequence)
    local_tokens = 0
    for i in range(num_seqs):
        seq_len = (cu_seqlens_global[i + 1] - cu_seqlens_global[i]).item()
        chunk_size = seq_len // (2 * world_size)
        local_tokens += 2 * chunk_size

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
        local_tokens_per_seq = 2 * (seq_len // (2 * world_size))

        grid = lambda META: (
            triton.cdiv(local_tokens_per_seq, BLOCK_SIZE),
            nheads_k,
            1,  # Process one sequence at a time
        )

        with torch.cuda.device(grad_contiguous.device.index):
            scatter_grad_to_zigzag_kernel[grid](
                grad_contiguous,
                grad_zigzag,
                cu_seqlens_global,
                rank,
                world_size,
                # Contiguous gradstrides
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
