"""
Tests for Fused Dual-Group Varlen Flash Attention

This test suite validates:
1. Single sequence correctness
2. Multi-sequence varlen correctness
3. Integration with execute_grouped_attention
4. Performance benchmark
"""

import pytest
import torch
import math
from typing import Tuple

# Import the fused kernel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ring_flash_attn.triton_fused_dual_group_flash_attn import (
    fused_zigzag_llama3_flash_attn_varlen_forward,
    replace_execute_grouped_attention_with_fused_varlen,
)
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_tensors(
    batch_size: int,
    seq_lens_q0: list,
    seq_lens_q1: list,
    seq_lens_k: list,
    nheads: int,
    nheads_k: int,
    headdim: int,
    dtype=torch.float16,
    device='cuda'
) -> Tuple:
    """Create test tensors in varlen format."""
    total_q0_tokens = sum(seq_lens_q0)
    total_q1_tokens = sum(seq_lens_q1)
    total_k_tokens = sum(seq_lens_k)

    q0 = torch.randn(total_q0_tokens, nheads, headdim, dtype=dtype, device=device)
    q1 = torch.randn(total_q1_tokens, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total_k_tokens, nheads_k, headdim, dtype=dtype, device=device)
    v = torch.randn(total_k_tokens, nheads_k, headdim, dtype=dtype, device=device)

    # Create cu_seqlens
    cu_seqlens_q0 = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens_q0), 0)),
                                  dtype=torch.int32, device=device)
    cu_seqlens_q1 = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens_q1), 0)),
                                  dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens_k), 0)),
                                 dtype=torch.int32, device=device)

    return q0, q1, k, v, cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k


# ============================================================================
# Test 1: Single Sequence Correctness
# ============================================================================

@pytest.mark.parametrize("seqlen_local", [128, 256])
@pytest.mark.parametrize("nheads", [8, 16])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("world_size", [2, 4])
def test_single_sequence_correctness(seqlen_local, nheads, headdim, world_size):
    """Test with batch_size=1, verify outputs match original."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size = 1
    rank = 0
    nheads_k = nheads  # No GQA for simplicity in this test

    # Create inputs
    seqlen_k_global = seqlen_local * world_size
    q0_len = seqlen_local // 2
    q1_len = seqlen_local // 2

    q0, q1, k, v, cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k = create_test_tensors(
        batch_size,
        [q0_len], [q1_len], [seqlen_k_global],
        nheads, nheads_k, headdim
    )

    # Compute attention ranges (zigzag pattern)
    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    total_chunks = 2 * world_size
    chunk_size = seqlen_k_global // total_chunks
    max_kv_len_q0 = (chunk_idx_0 + 1) * chunk_size
    max_kv_len_q1 = (chunk_idx_1 + 1) * chunk_size

    softmax_scale = 1.0 / math.sqrt(headdim)

    # Run fused kernel
    out0_fused, out1_fused, lse0_fused, lse1_fused = fused_zigzag_llama3_flash_attn_varlen_forward(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        q0_len, q1_len, seqlen_k_global,
        max_kv_len_q0, max_kv_len_q1,
        softmax_scale, causal=True,
    )

    # Run original (two separate calls)
    # For Q0
    k0 = k[:max_kv_len_q0]
    v0 = v[:max_kv_len_q0]
    cu_seqlens_k0 = torch.tensor([0, max_kv_len_q0], dtype=torch.int32, device='cuda')

    out0_orig, _, _, _, _, lse0_orig, _, _ = _flash_attn_varlen_forward(
        q0, k0, v0,
        cu_seqlens_q0, cu_seqlens_k0,
        q0_len, max_kv_len_q0,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # For Q1
    k1 = k[:max_kv_len_q1]
    v1 = v[:max_kv_len_q1]
    cu_seqlens_k1 = torch.tensor([0, max_kv_len_q1], dtype=torch.int32, device='cuda')

    out1_orig, _, _, _, _, lse1_orig, _, _ = _flash_attn_varlen_forward(
        q1, k1, v1,
        cu_seqlens_q1, cu_seqlens_k1,
        q1_len, max_kv_len_q1,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # Compare outputs
    atol, rtol = 1e-2, 1e-2

    print(f"\nTest: seqlen={seqlen_local}, nheads={nheads}, headdim={headdim}, world_size={world_size}")
    print(f"Q0 output diff: max={torch.max(torch.abs(out0_fused - out0_orig)).item():.6f}, "
          f"mean={torch.mean(torch.abs(out0_fused - out0_orig)).item():.6f}")
    print(f"Q1 output diff: max={torch.max(torch.abs(out1_fused - out1_orig)).item():.6f}, "
          f"mean={torch.mean(torch.abs(out1_fused - out1_orig)).item():.6f}")

    assert torch.allclose(out0_fused, out0_orig, atol=atol, rtol=rtol), \
        f"Q0 outputs don't match! Max diff: {torch.max(torch.abs(out0_fused - out0_orig))}"
    assert torch.allclose(out1_fused, out1_orig, atol=atol, rtol=rtol), \
        f"Q1 outputs don't match! Max diff: {torch.max(torch.abs(out1_fused - out1_orig))}"

    print("✓ Single sequence test passed")


# ============================================================================
# Test 2: Multi-Sequence Varlen Correctness
# ============================================================================

@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("headdim", [64, 128])
def test_multi_sequence_varlen_correctness(batch_size, nheads, headdim):
    """Test with multiple sequences of different lengths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nheads_k = nheads

    # Different sequence lengths
    if batch_size == 2:
        seq_lens_q0 = [30, 80]
        seq_lens_q1 = [70, 120]
        seq_lens_k = [400, 800]
    else:  # batch_size == 3
        seq_lens_q0 = [30, 80, 50]
        seq_lens_q1 = [70, 120, 90]
        seq_lens_k = [400, 800, 600]

    q0, q1, k, v, cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k = create_test_tensors(
        batch_size, seq_lens_q0, seq_lens_q1, seq_lens_k,
        nheads, nheads_k, headdim
    )

    # Test with different max_kv_len for each group
    max_kv_len_q0 = 100
    max_kv_len_q1 = max(seq_lens_k)

    max_seqlen_q0 = max(seq_lens_q0)
    max_seqlen_q1 = max(seq_lens_q1)
    max_seqlen_k = max(seq_lens_k)

    softmax_scale = 1.0 / math.sqrt(headdim)

    # Run fused kernel
    out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
        max_kv_len_q0, max_kv_len_q1,
        softmax_scale, causal=True,
    )

    # Verify per-sequence by calling original on each sequence
    atol, rtol = 1e-2, 1e-2

    for i in range(batch_size):
        # Extract sequences
        q0_seq = q0[cu_seqlens_q0[i]:cu_seqlens_q0[i+1]]
        q1_seq = q1[cu_seqlens_q1[i]:cu_seqlens_q1[i+1]]
        k_seq = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
        v_seq = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]]

        seq_len_q0 = seq_lens_q0[i]
        seq_len_q1 = seq_lens_q1[i]
        seq_len_k = seq_lens_k[i]

        # Test Q0
        k0_range = min(max_kv_len_q0, seq_len_k)
        cu_seqlens_q0_seq = torch.tensor([0, seq_len_q0], dtype=torch.int32, device='cuda')
        cu_seqlens_k0_seq = torch.tensor([0, k0_range], dtype=torch.int32, device='cuda')

        out0_expected, _, _, _, _, _, _, _ = _flash_attn_varlen_forward(
            q0_seq, k_seq[:k0_range], v_seq[:k0_range],
            cu_seqlens_q0_seq, cu_seqlens_k0_seq,
            seq_len_q0, k0_range,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
        )

        out0_actual = out0[cu_seqlens_q0[i]:cu_seqlens_q0[i+1]]

        print(f"\nSeq {i}, Q0: max diff={torch.max(torch.abs(out0_actual - out0_expected)).item():.6f}")
        assert torch.allclose(out0_actual, out0_expected, atol=atol, rtol=rtol), \
            f"Q0 seq {i} mismatch! Max diff: {torch.max(torch.abs(out0_actual - out0_expected))}"

        # Test Q1
        k1_range = min(max_kv_len_q1, seq_len_k)
        cu_seqlens_q1_seq = torch.tensor([0, seq_len_q1], dtype=torch.int32, device='cuda')
        cu_seqlens_k1_seq = torch.tensor([0, k1_range], dtype=torch.int32, device='cuda')

        out1_expected, _, _, _, _, _, _, _ = _flash_attn_varlen_forward(
            q1_seq, k_seq[:k1_range], v_seq[:k1_range],
            cu_seqlens_q1_seq, cu_seqlens_k1_seq,
            seq_len_q1, k1_range,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
        )

        out1_actual = out1[cu_seqlens_q1[i]:cu_seqlens_q1[i+1]]

        print(f"Seq {i}, Q1: max diff={torch.max(torch.abs(out1_actual - out1_expected)).item():.6f}")
        assert torch.allclose(out1_actual, out1_expected, atol=atol, rtol=rtol), \
            f"Q1 seq {i} mismatch! Max diff: {torch.max(torch.abs(out1_actual - out1_expected))}"

    print("✓ Multi-sequence varlen test passed")


# ============================================================================
# Test 3: Integration Test (Drop-in Replacement)
# ============================================================================

def test_integration_with_execute_grouped_attention():
    """Test drop-in replacement for execute_grouped_attention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from ring_flash_attn.zigzag_llama3_flash_attn_varlen import (
        split_q_by_zigzag_chunk_index,
        compute_kv_slices_for_groups,
        execute_grouped_attention,
        rearrange_kv_from_zigzag_to_contiguous,
    )

    world_size = 4
    rank = 0
    seqlen_local = 256
    nheads = 8
    nheads_k = 8
    headdim = 64
    batch_size = 1

    # Create test data in zigzag interleaved format
    # For simplicity, we'll create local Q (interleaved) and replicate K,V
    q_local = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device='cuda')
    k_local = torch.randn(seqlen_local, nheads_k, headdim, dtype=torch.float16, device='cuda')
    v_local = torch.randn(seqlen_local, nheads_k, headdim, dtype=torch.float16, device='cuda')

    # Create KV buffer (simulate all-gather by replicating)
    kv_buffer = torch.stack([
        k_local.repeat(world_size, 1, 1),
        v_local.repeat(world_size, 1, 1)
    ])

    cu_seqlens_q = torch.tensor([0, seqlen_local], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, seqlen_local], dtype=torch.int32, device='cuda')

    # Prepare inputs using original functions
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q_local, cu_seqlens_q, world_size, rank, n_chunks=2
    )

    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    kv_slices = compute_kv_slices_for_groups(
        cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size,
        chunk_cu_seqlens_q_list, n_chunks=2
    )

    softmax_scale = 1.0 / math.sqrt(headdim)

    # Run original
    out_orig, lse_orig, _ = execute_grouped_attention(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k
    )

    # Run fused
    out_fused, lse_fused, _ = replace_execute_grouped_attention_with_fused_varlen(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k,
        chunk_idx_0=chunk_idx_0, chunk_idx_1=chunk_idx_1
    )

    # Compare
    atol, rtol = 1e-2, 1e-2

    print(f"\nIntegration test:")
    print(f"Output diff: max={torch.max(torch.abs(out_fused - out_orig)).item():.6f}, "
          f"mean={torch.mean(torch.abs(out_fused - out_orig)).item():.6f}")
    print(f"LSE diff: max={torch.max(torch.abs(lse_fused - lse_orig)).item():.6f}, "
          f"mean={torch.mean(torch.abs(lse_fused - lse_orig)).item():.6f}")

    assert torch.allclose(out_fused, out_orig, atol=atol, rtol=rtol), \
        f"Outputs don't match! Max diff: {torch.max(torch.abs(out_fused - out_orig))}"
    assert torch.allclose(lse_fused, lse_orig, atol=atol, rtol=rtol), \
        f"LSE doesn't match! Max diff: {torch.max(torch.abs(lse_fused - lse_orig))}"

    print("✓ Integration test passed")


# ============================================================================
# Test 4: Performance Benchmark
# ============================================================================

@pytest.mark.benchmark
def test_performance_comparison():
    """Benchmark fused vs original."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from ring_flash_attn.zigzag_llama3_flash_attn_varlen import (
        split_q_by_zigzag_chunk_index,
        compute_kv_slices_for_groups,
        execute_grouped_attention,
    )

    configs = [
        (256, 8, 64),
        (512, 16, 64),
        (512, 32, 128),
    ]

    world_size = 4
    rank = 0
    num_iters = 50

    print("\n" + "="*80)
    print("Performance Benchmark: Fused vs Original")
    print("="*80)

    for seqlen, nheads, headdim in configs:
        nheads_k = nheads
        seqlen_k_global = seqlen * world_size

        # Setup
        q_local = torch.randn(seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
        k_local = torch.randn(seqlen, nheads_k, headdim, dtype=torch.float16, device='cuda')
        v_local = torch.randn(seqlen, nheads_k, headdim, dtype=torch.float16, device='cuda')

        kv_buffer = torch.stack([
            k_local.repeat(world_size, 1, 1),
            v_local.repeat(world_size, 1, 1)
        ])

        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device='cuda')
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device='cuda')

        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
            q_local, cu_seqlens_q, world_size, rank, n_chunks=2
        )

        chunk_idx_0 = rank
        chunk_idx_1 = 2 * world_size - 1 - rank
        kv_slices = compute_kv_slices_for_groups(
            cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size,
            chunk_cu_seqlens_q_list, n_chunks=2
        )

        softmax_scale = 1.0 / math.sqrt(headdim)

        # Warmup
        for _ in range(5):
            _ = execute_grouped_attention(
                chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads, headdim, softmax_scale,
                dropout_p=0.0, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                world_size=world_size, cu_seqlens_k=cu_seqlens_k
            )
            _ = replace_execute_grouped_attention_with_fused_varlen(
                chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads, headdim, softmax_scale,
                dropout_p=0.0, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                world_size=world_size, cu_seqlens_k=cu_seqlens_k,
                chunk_idx_0=chunk_idx_0, chunk_idx_1=chunk_idx_1
            )

        # Benchmark original
        torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(num_iters):
            out_orig, _, _ = execute_grouped_attention(
                chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads, headdim, softmax_scale,
                dropout_p=0.0, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                world_size=world_size, cu_seqlens_k=cu_seqlens_k
            )
        torch.cuda.synchronize()
        time_orig = (time.time() - start) / num_iters * 1000  # ms

        # Benchmark fused
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            out_fused, _, _ = replace_execute_grouped_attention_with_fused_varlen(
                chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
                kv_buffer, kv_slices,
                nheads, headdim, softmax_scale,
                dropout_p=0.0, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                world_size=world_size, cu_seqlens_k=cu_seqlens_k,
                chunk_idx_0=chunk_idx_0, chunk_idx_1=chunk_idx_1
            )
        torch.cuda.synchronize()
        time_fused = (time.time() - start) / num_iters * 1000  # ms

        speedup = time_orig / time_fused

        print(f"\nConfig: seqlen={seqlen}, nheads={nheads}, headdim={headdim}")
        print(f"  Original: {time_orig:.3f} ms")
        print(f"  Fused:    {time_fused:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")

        # We expect at least some speedup, but the exact amount depends on hardware
        # For now, just verify it runs without errors
        assert speedup > 0.8, f"Fused kernel is slower! Speedup: {speedup:.2f}x"

    print("\n" + "="*80)
    print("✓ Performance benchmark completed")
    print("="*80)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("Running Fused Dual-Group Varlen Flash Attention Tests\n")

    # Test 1: Single sequence correctness
    print("\n" + "="*80)
    print("Test 1: Single Sequence Correctness")
    print("="*80)
    for seqlen in [128, 256]:
        for nheads in [8, 16]:
            for headdim in [64, 128]:
                for world_size in [2, 4]:
                    test_single_sequence_correctness(seqlen, nheads, headdim, world_size)

    # Test 2: Multi-sequence varlen correctness
    print("\n" + "="*80)
    print("Test 2: Multi-Sequence Varlen Correctness")
    print("="*80)
    for batch_size in [2, 3]:
        for nheads in [8]:
            for headdim in [64, 128]:
                test_multi_sequence_varlen_correctness(batch_size, nheads, headdim)

    # Test 3: Integration test
    print("\n" + "="*80)
    print("Test 3: Integration Test (Drop-in Replacement)")
    print("="*80)
    test_integration_with_execute_grouped_attention()

    # Test 4: Performance benchmark
    print("\n" + "="*80)
    print("Test 4: Performance Benchmark")
    print("="*80)
    test_performance_comparison()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
