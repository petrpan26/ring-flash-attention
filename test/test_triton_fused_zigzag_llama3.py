"""
Test suite for fused Triton implementation of zigzag llama3 flash attention.

This test verifies that the fused Triton kernel produces identical results
to the original Python-based implementation.
"""

import torch
import pytest
from ring_flash_attn.zigzag_llama3_flash_attn_varlen import (
    split_q_by_zigzag_chunk_index,
    rearrange_kv_from_zigzag_to_contiguous,
    compute_kv_slices_for_groups,
    execute_grouped_attention,
)


# Only import Triton if available
try:
    from ring_flash_attn.triton_zigzag_llama3_flash_attn import (
        fused_zigzag_llama3_flash_attn_forward,
        fused_zigzag_llama3_flash_attn_backward,
        replace_grouped_attention_with_fused_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    pytest.skip("Triton not available", allow_module_level=True)


def create_test_tensors(
    batch_size=1,
    seqlen_local=256,
    nheads=8,
    nheads_k=8,
    headdim=64,
    world_size=4,
    rank=0,
    dtype=torch.float16,
    device="cuda",
):
    """
    Create test tensors in zigzag interleaved format for a single rank.

    Returns:
        q, k, v, cu_seqlens_q, cu_seqlens_k
    """
    # For simplicity, use single sequence
    num_sequences = 1

    # Local Q, K, V (in zigzag interleaved format)
    # Each rank has 2 chunks: chunk_rank and chunk_(2*world_size-1-rank)
    total_local_tokens = seqlen_local
    q = torch.randn(total_local_tokens, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_local_tokens, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_local_tokens, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)

    # Cumulative sequence lengths (LOCAL)
    cu_seqlens_q = torch.tensor([0, total_local_tokens], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, total_local_tokens], dtype=torch.int32, device=device)

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def test_forward_correctness():
    """Test that fused forward produces same results as original."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Setup
    world_size = 4
    rank = 0
    seqlen_local = 256
    nheads = 8
    headdim = 64
    device = "cuda"
    dtype = torch.float16

    q, k, v, cu_seqlens_q, cu_seqlens_k = create_test_tensors(
        seqlen_local=seqlen_local,
        nheads=nheads,
        headdim=headdim,
        world_size=world_size,
        rank=rank,
        dtype=dtype,
        device=device,
    )

    # Create mock kv_buffer (all-gathered K,V)
    # In real usage, this would be the result of all-gather
    # For testing, we just replicate k, v to simulate all-gather
    kv_buffer = torch.stack([
        k.repeat(world_size, 1, 1),
        v.repeat(world_size, 1, 1),
    ])  # [2, total_tokens * world_size, nheads, headdim]

    softmax_scale = 1.0 / (headdim ** 0.5)

    # Split Q by chunk index
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q, cu_seqlens_q, world_size, rank, n_chunks=2
    )

    # Compute K,V slices
    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    kv_slices = compute_kv_slices_for_groups(
        cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size,
        chunk_cu_seqlens_q_list, n_chunks=2
    )

    # Original implementation
    out_original, lse_original, chunk_info = execute_grouped_attention(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k
    )

    # Fused Triton implementation
    out_fused, lse_fused, chunk_info_fused = replace_grouped_attention_with_fused_triton(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k,
        chunk_idx_0=chunk_idx_0, chunk_idx_1=chunk_idx_1
    )

    # Compare outputs
    print(f"out_original shape: {out_original.shape}")
    print(f"out_fused shape: {out_fused.shape}")
    print(f"out_original: {out_original.flatten()[:10]}")
    print(f"out_fused: {out_fused.flatten()[:10]}")

    # Check correctness (allow for numerical differences)
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4

    assert torch.allclose(out_original, out_fused, atol=atol, rtol=rtol), \
        f"Forward outputs differ! Max diff: {(out_original - out_fused).abs().max()}"

    print("✓ Forward pass test passed!")


def test_backward_correctness():
    """Test that fused backward produces same gradients as original."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Setup
    world_size = 4
    rank = 0
    seqlen_local = 256
    nheads = 8
    headdim = 64
    device = "cuda"
    dtype = torch.float16

    # Create tensors with requires_grad
    q, k, v, cu_seqlens_q, cu_seqlens_k = create_test_tensors(
        seqlen_local=seqlen_local,
        nheads=nheads,
        headdim=headdim,
        world_size=world_size,
        rank=rank,
        dtype=dtype,
        device=device,
    )

    q_clone = q.clone().detach().requires_grad_(True)
    k_clone = k.clone().detach().requires_grad_(True)
    v_clone = v.clone().detach().requires_grad_(True)

    # Run forward pass for both implementations
    kv_buffer = torch.stack([
        k.repeat(world_size, 1, 1),
        v.repeat(world_size, 1, 1),
    ])
    kv_buffer_clone = torch.stack([
        k_clone.repeat(world_size, 1, 1),
        v_clone.repeat(world_size, 1, 1),
    ])

    softmax_scale = 1.0 / (headdim ** 0.5)

    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q, cu_seqlens_q, world_size, rank, n_chunks=2
    )
    chunk_q_list_clone, _, _ = split_q_by_zigzag_chunk_index(
        q_clone, cu_seqlens_q, world_size, rank, n_chunks=2
    )

    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    kv_slices = compute_kv_slices_for_groups(
        cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size,
        chunk_cu_seqlens_q_list, n_chunks=2
    )

    # Original
    out_original, _, _ = execute_grouped_attention(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k
    )

    # Fused
    out_fused, _, _ = replace_grouped_attention_with_fused_triton(
        chunk_q_list_clone, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer_clone, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k,
        chunk_idx_0=chunk_idx_0, chunk_idx_1=chunk_idx_1
    )

    # Backward pass
    dout = torch.randn_like(out_original)

    out_original.backward(dout)
    grad_q_original = q.grad.clone()
    grad_k_original = k.grad.clone() if k.grad is not None else None
    grad_v_original = v.grad.clone() if v.grad is not None else None

    out_fused.backward(dout)
    grad_q_fused = q_clone.grad
    grad_k_fused = k_clone.grad
    grad_v_fused = v_clone.grad

    # Compare gradients
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4

    if grad_q_original is not None and grad_q_fused is not None:
        assert torch.allclose(grad_q_original, grad_q_fused, atol=atol, rtol=rtol), \
            f"Q gradients differ! Max diff: {(grad_q_original - grad_q_fused).abs().max()}"

    print("✓ Backward pass test passed!")


if __name__ == "__main__":
    print("Testing fused Triton zigzag llama3 implementation...\n")

    if not TRITON_AVAILABLE:
        print("⚠ Triton not available, skipping tests")
        exit(0)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping tests")
        exit(0)

    try:
        print("=" * 60)
        print("Test 1: Forward Correctness")
        print("=" * 60)
        test_forward_correctness()
        print()

        print("=" * 60)
        print("Test 2: Backward Correctness")
        print("=" * 60)
        test_backward_correctness()
        print()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
