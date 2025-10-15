"""
Unit tests for Triton kernel optimizations in zigzag_llama3.

Tests the following kernels:
1. extract_zigzag_kv_slice_kernel - Direct KV extraction from zigzag format
2. scatter_grad_to_zigzag_kernel - Gradient scattering for backward pass

Each test compares Triton kernel output against the Python reference implementation.
"""

import sys
import torch
import torch.distributed as dist
import pytest

# Add parent directory to path
sys.path.insert(0, '/Users/petrpan26/work/ring-flash-attention')

from ring_flash_attn.triton_utils import extract_zigzag_kv_slices_for_group, scatter_grad_to_zigzag
from ring_flash_attn.zigzag_llama3_flash_attn_varlen import rearrange_kv_from_zigzag_to_contiguous, rearrange_grad_from_contiguous_to_zigzag


def setup_distributed():
    """Initialize distributed environment for testing."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    return rank, world_size, device


def teardown_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Test #1: extract_zigzag_kv_slice_kernel
# ============================================================================

def test_extract_zigzag_kv_slices_single_sequence():
    """Test Triton KV extraction kernel with a single sequence."""
    rank, world_size, device = setup_distributed()

    # Configuration
    seq_len = 1024  # Must be divisible by 2*world_size
    nheads_k = 8
    head_dim = 128
    dtype = torch.bfloat16

    assert seq_len % (2 * world_size) == 0, "seq_len must be divisible by 2*world_size"

    # Create zigzag interleaved KV buffer (simulating all-gather output)
    total_tokens = seq_len
    local_tokens_per_rank = seq_len // world_size
    kv_buffer = torch.randn(
        2, total_tokens, nheads_k, head_dim,
        device=device, dtype=dtype
    )

    # Global cu_seqlens
    cu_seqlens_global = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # Test extracting chunks for early group (chunks 0 to world_size-1)
    chunk_size = seq_len // (2 * world_size)
    max_chunk_idx = world_size - 1  # Extract chunks [0, 1, ..., world_size-1]
    num_tokens_needed = (max_chunk_idx + 1) * chunk_size

    seq_ranges_with_chunk_idx = [(0, num_tokens_needed, max_chunk_idx)]

    # Method 1: Triton kernel (OPTIMIZED)
    k_triton, v_triton = extract_zigzag_kv_slices_for_group(
        kv_buffer,
        seq_ranges_with_chunk_idx,
        cu_seqlens_global,
        world_size,
        nheads_k,
        head_dim,
    )

    # Method 2: Python reference (BASELINE)
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_global)
    k_python = kv_contiguous[0, 0:num_tokens_needed]
    v_python = kv_contiguous[1, 0:num_tokens_needed]

    # Compare
    k_diff = (k_triton - k_python).abs()
    v_diff = (v_triton - v_python).abs()

    if rank == 0:
        print(f"\n[Rank {rank}] Test: extract_zigzag_kv_slices_single_sequence")
        print(f"  K max diff: {k_diff.max().item():.6e}")
        print(f"  K mean diff: {k_diff.mean().item():.6e}")
        print(f"  V max diff: {v_diff.max().item():.6e}")
        print(f"  V mean diff: {v_diff.mean().item():.6e}")

    # Should be exactly equal (no numerical errors in indexing)
    assert torch.allclose(k_triton, k_python, rtol=0, atol=0), \
        f"K mismatch: max diff = {k_diff.max().item()}"
    assert torch.allclose(v_triton, v_python, rtol=0, atol=0), \
        f"V mismatch: max diff = {v_diff.max().item()}"

    if rank == 0:
        print("  ✓ PASSED")


def test_extract_zigzag_kv_slices_multiple_sequences():
    """Test Triton KV extraction kernel with multiple sequences."""
    rank, world_size, device = setup_distributed()

    # Configuration
    seq_lens = [512, 1024, 768]  # Different lengths
    nheads_k = 4
    head_dim = 64
    dtype = torch.float16

    # Ensure all sequences divisible by 2*world_size
    for seq_len in seq_lens:
        assert seq_len % (2 * world_size) == 0

    # Create global cu_seqlens
    cu_seqlens_list = [0]
    for seq_len in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + seq_len)
    cu_seqlens_global = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

    total_tokens = cu_seqlens_global[-1].item()
    local_tokens_per_rank = total_tokens // world_size

    # Create zigzag KV buffer
    kv_buffer = torch.randn(
        2, total_tokens, nheads_k, head_dim,
        device=device, dtype=dtype
    )

    # Build seq_ranges for extracting all sequences (early group only)
    seq_ranges_with_chunk_idx = []
    for i, seq_len in enumerate(seq_lens):
        chunk_size = seq_len // (2 * world_size)
        max_chunk_idx = world_size - 1  # Early chunks only
        num_tokens = (max_chunk_idx + 1) * chunk_size

        seq_start = cu_seqlens_global[i].item()
        seq_end = seq_start + num_tokens
        seq_ranges_with_chunk_idx.append((seq_start, seq_end, max_chunk_idx))

    # Method 1: Triton kernel
    k_triton, v_triton = extract_zigzag_kv_slices_for_group(
        kv_buffer,
        seq_ranges_with_chunk_idx,
        cu_seqlens_global,
        world_size,
        nheads_k,
        head_dim,
    )

    # Method 2: Python reference
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_global)

    # Extract same slices from Python version
    k_slices = []
    v_slices = []
    for start, end, _ in seq_ranges_with_chunk_idx:
        k_slices.append(kv_contiguous[0, start:end])
        v_slices.append(kv_contiguous[1, start:end])
    k_python = torch.cat(k_slices, dim=0)
    v_python = torch.cat(v_slices, dim=0)

    # Compare
    k_diff = (k_triton - k_python).abs()
    v_diff = (v_triton - v_python).abs()

    if rank == 0:
        print(f"\n[Rank {rank}] Test: extract_zigzag_kv_slices_multiple_sequences")
        print(f"  Sequences: {seq_lens}")
        print(f"  K max diff: {k_diff.max().item():.6e}")
        print(f"  V max diff: {v_diff.max().item():.6e}")

    assert torch.allclose(k_triton, k_python, rtol=0, atol=0)
    assert torch.allclose(v_triton, v_python, rtol=0, atol=0)

    if rank == 0:
        print("  ✓ PASSED")


def test_extract_zigzag_kv_slices_late_chunks():
    """Test Triton KV extraction kernel with late chunks (world_size to 2*world_size-1)."""
    rank, world_size, device = setup_distributed()

    # Configuration
    seq_len = 2048
    nheads_k = 8
    head_dim = 128
    dtype = torch.bfloat16

    assert seq_len % (2 * world_size) == 0

    # Create KV buffer
    total_tokens = seq_len
    kv_buffer = torch.randn(
        2, total_tokens, nheads_k, head_dim,
        device=device, dtype=dtype
    )

    cu_seqlens_global = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # Extract LATE chunks only (chunks world_size to 2*world_size-1)
    chunk_size = seq_len // (2 * world_size)
    start_chunk_idx = world_size
    end_chunk_idx = 2 * world_size - 1

    # For causal attention, we need chunks [0, 1, ..., end_chunk_idx]
    max_chunk_idx = end_chunk_idx
    num_tokens_needed = (max_chunk_idx + 1) * chunk_size

    seq_ranges_with_chunk_idx = [(0, num_tokens_needed, max_chunk_idx)]

    # Method 1: Triton kernel
    k_triton, v_triton = extract_zigzag_kv_slices_for_group(
        kv_buffer,
        seq_ranges_with_chunk_idx,
        cu_seqlens_global,
        world_size,
        nheads_k,
        head_dim,
    )

    # Method 2: Python reference
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_global)
    k_python = kv_contiguous[0, 0:num_tokens_needed]
    v_python = kv_contiguous[1, 0:num_tokens_needed]

    # Compare
    k_diff = (k_triton - k_python).abs()
    v_diff = (v_triton - v_python).abs()

    if rank == 0:
        print(f"\n[Rank {rank}] Test: extract_zigzag_kv_slices_late_chunks")
        print(f"  Extracting chunks [0..{max_chunk_idx}] (includes late chunks)")
        print(f"  K max diff: {k_diff.max().item():.6e}")
        print(f"  V max diff: {v_diff.max().item():.6e}")

    assert torch.allclose(k_triton, k_python, rtol=0, atol=0)
    assert torch.allclose(v_triton, v_python, rtol=0, atol=0)

    if rank == 0:
        print("  ✓ PASSED")


# ============================================================================
# Test #2: scatter_grad_to_zigzag_kernel
# ============================================================================

def test_scatter_grad_to_zigzag_single_sequence():
    """Test Triton gradient scatter kernel with a single sequence."""
    rank, world_size, device = setup_distributed()

    # Configuration
    seq_len = 1024
    nheads_k = 8
    head_dim = 128
    dtype = torch.bfloat16

    assert seq_len % (2 * world_size) == 0

    # Create contiguous gradient (simulating accumulated backward gradients)
    grad_contiguous = torch.randn(
        2, seq_len, nheads_k, head_dim,
        device=device, dtype=dtype
    )

    cu_seqlens_global = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # Method 1: Triton kernel (OPTIMIZED)
    # Note: This returns only THIS rank's portion
    grad_zigzag_triton_dk = scatter_grad_to_zigzag(
        grad_contiguous[0],  # K gradients
        cu_seqlens_global,
        rank,
        world_size,
    )
    grad_zigzag_triton_dv = scatter_grad_to_zigzag(
        grad_contiguous[1],  # V gradients
        cu_seqlens_global,
        rank,
        world_size,
    )

    # Method 2: Python reference (BASELINE)
    # This returns ALL ranks' portions concatenated
    grad_zigzag_python = rearrange_grad_from_contiguous_to_zigzag(
        grad_contiguous,
        world_size,
        cu_seqlens_global,
    )

    # Extract this rank's portion from Python result
    local_tokens = seq_len // world_size
    start = rank * local_tokens
    end = start + local_tokens
    grad_python_dk = grad_zigzag_python[0, start:end]
    grad_python_dv = grad_zigzag_python[1, start:end]

    # Compare
    dk_diff = (grad_zigzag_triton_dk - grad_python_dk).abs()
    dv_diff = (grad_zigzag_triton_dv - grad_python_dv).abs()

    if rank == 0:
        print(f"\n[Rank {rank}] Test: scatter_grad_to_zigzag_single_sequence")
        print(f"  dK max diff: {dk_diff.max().item():.6e}")
        print(f"  dK mean diff: {dk_diff.mean().item():.6e}")
        print(f"  dV max diff: {dv_diff.max().item():.6e}")
        print(f"  dV mean diff: {dv_diff.mean().item():.6e}")

    # Should be exactly equal
    assert torch.allclose(grad_zigzag_triton_dk, grad_python_dk, rtol=0, atol=0), \
        f"dK mismatch: max diff = {dk_diff.max().item()}"
    assert torch.allclose(grad_zigzag_triton_dv, grad_python_dv, rtol=0, atol=0), \
        f"dV mismatch: max diff = {dv_diff.max().item()}"

    if rank == 0:
        print("  ✓ PASSED")


def test_scatter_grad_to_zigzag_multiple_sequences():
    """Test Triton gradient scatter kernel with multiple sequences."""
    rank, world_size, device = setup_distributed()

    # Configuration
    seq_lens = [512, 1024, 768]
    nheads_k = 4
    head_dim = 64
    dtype = torch.float16

    for seq_len in seq_lens:
        assert seq_len % (2 * world_size) == 0

    # Create global cu_seqlens
    cu_seqlens_list = [0]
    for seq_len in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + seq_len)
    cu_seqlens_global = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

    total_tokens = cu_seqlens_global[-1].item()

    # Create contiguous gradient
    grad_contiguous = torch.randn(
        2, total_tokens, nheads_k, head_dim,
        device=device, dtype=dtype
    )

    # Method 1: Triton kernel
    grad_zigzag_triton_dk = scatter_grad_to_zigzag(
        grad_contiguous[0],
        cu_seqlens_global,
        rank,
        world_size,
    )
    grad_zigzag_triton_dv = scatter_grad_to_zigzag(
        grad_contiguous[1],
        cu_seqlens_global,
        rank,
        world_size,
    )

    # Method 2: Python reference
    grad_zigzag_python = rearrange_grad_from_contiguous_to_zigzag(
        grad_contiguous,
        world_size,
        cu_seqlens_global,
    )

    # Extract this rank's portion
    local_tokens = total_tokens // world_size
    start = rank * local_tokens
    end = start + local_tokens
    grad_python_dk = grad_zigzag_python[0, start:end]
    grad_python_dv = grad_zigzag_python[1, start:end]

    # Compare
    dk_diff = (grad_zigzag_triton_dk - grad_python_dk).abs()
    dv_diff = (grad_zigzag_triton_dv - grad_python_dv).abs()

    if rank == 0:
        print(f"\n[Rank {rank}] Test: scatter_grad_to_zigzag_multiple_sequences")
        print(f"  Sequences: {seq_lens}")
        print(f"  dK max diff: {dk_diff.max().item():.6e}")
        print(f"  dV max diff: {dv_diff.max().item():.6e}")

    assert torch.allclose(grad_zigzag_triton_dk, grad_python_dk, rtol=0, atol=0)
    assert torch.allclose(grad_zigzag_triton_dv, grad_python_dv, rtol=0, atol=0)

    if rank == 0:
        print("  ✓ PASSED")


# ============================================================================
# Test #3: Performance Comparison
# ============================================================================

def test_triton_kernel_performance():
    """Benchmark Triton kernel vs Python reference."""
    rank, world_size, device = setup_distributed()

    if rank != 0:
        return  # Only benchmark on rank 0

    # Configuration
    seq_len = 8192
    nheads_k = 32
    head_dim = 128
    dtype = torch.bfloat16
    num_warmup = 5
    num_iter = 20

    assert seq_len % (2 * world_size) == 0

    # Create data
    total_tokens = seq_len
    kv_buffer = torch.randn(
        2, total_tokens, nheads_k, head_dim,
        device=device, dtype=dtype
    )
    cu_seqlens_global = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    chunk_size = seq_len // (2 * world_size)
    max_chunk_idx = 2 * world_size - 1  # All chunks
    num_tokens_needed = (max_chunk_idx + 1) * chunk_size
    seq_ranges_with_chunk_idx = [(0, num_tokens_needed, max_chunk_idx)]

    # Warmup
    for _ in range(num_warmup):
        k_triton, v_triton = extract_zigzag_kv_slices_for_group(
            kv_buffer, seq_ranges_with_chunk_idx, cu_seqlens_global,
            world_size, nheads_k, head_dim
        )
        kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(
            kv_buffer, world_size, cu_seqlens_global
        )

    torch.cuda.synchronize()

    # Benchmark Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iter):
        k_triton, v_triton = extract_zigzag_kv_slices_for_group(
            kv_buffer, seq_ranges_with_chunk_idx, cu_seqlens_global,
            world_size, nheads_k, head_dim
        )
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / num_iter

    # Benchmark Python
    start.record()
    for _ in range(num_iter):
        kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(
            kv_buffer, world_size, cu_seqlens_global
        )
    end.record()
    torch.cuda.synchronize()
    python_time = start.elapsed_time(end) / num_iter

    speedup = python_time / triton_time

    print(f"\n[Rank {rank}] Performance Comparison:")
    print(f"  Triton kernel: {triton_time:.3f} ms")
    print(f"  Python reference: {python_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Should be faster
    assert speedup > 1.0, f"Triton kernel slower than Python! Speedup: {speedup:.2f}x"
    print("  ✓ PASSED (Triton is faster)")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    setup_distributed()

    try:
        # Test extract_zigzag_kv_slice_kernel
        print("\n" + "="*70)
        print("Testing: extract_zigzag_kv_slice_kernel")
        print("="*70)

        test_extract_zigzag_kv_slices_single_sequence()
        test_extract_zigzag_kv_slices_multiple_sequences()
        test_extract_zigzag_kv_slices_late_chunks()

        # Test scatter_grad_to_zigzag_kernel
        print("\n" + "="*70)
        print("Testing: scatter_grad_to_zigzag_kernel")
        print("="*70)

        test_scatter_grad_to_zigzag_single_sequence()
        test_scatter_grad_to_zigzag_multiple_sequences()

        # Performance test
        print("\n" + "="*70)
        print("Performance Benchmarks")
        print("="*70)

        test_triton_kernel_performance()

        # Summary
        rank = dist.get_rank()
        if rank == 0:
            print("\n" + "="*70)
            print("ALL TESTS PASSED! ✓")
            print("="*70)

    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
