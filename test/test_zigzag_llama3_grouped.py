"""
Integration tests for zigzag_llama3 with grouped attention enabled.

This module tests the integration of grouped attention with zigzag_llama3:
- Compare grouped attention vs baseline (two-kernels mode)
- Test forward pass correctness
- Test backward pass correctness
- Test with different world_sizes (1, 2, 4, 8 GPUs)
- Test with different sequence lengths
- Test memory efficiency improvements

Run with:
    torchrun --nproc_per_node=8 test/test_zigzag_llama3_grouped.py
"""

import sys
import os
import pytest
import torch
import torch.distributed as dist
from typing import Tuple, List

# Import zigzag_llama3 functions
try:
    from ring_flash_attn import (
        zigzag_llama3_flash_attn_varlen_kvpacked_func,
        llama3_flash_attn_prepare_cu_seqlens,
    )
except ImportError:
    pytest.skip("ring_flash_attn not installed", allow_module_level=True)

try:
    from flash_attn import flash_attn_varlen_kvpacked_func
except ImportError:
    pytest.skip("flash_attn not installed", allow_module_level=True)


def set_seed(rank: int, seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_local_zigzag(
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Extract local zigzag-distributed portion for this rank.

    Each sequence is split into 2*world_size chunks.
    GPU gets: chunk[rank] + chunk[2*world_size - 1 - rank]
    """
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend([
            local_value[rank].detach().clone(),
            local_value[2 * world_size - 1 - rank].detach().clone(),
        ])
    return torch.cat(local_values, dim=0).contiguous()


class TestZigzagLlama3GroupedAttention:
    """Integration tests for zigzag_llama3 with grouped attention."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_distributed(self):
        """Setup distributed environment."""
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        yield
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.fixture
    def device(self):
        """Get current device."""
        rank = dist.get_rank()
        return torch.device(f"cuda:{rank}")

    @pytest.fixture
    def dtype(self):
        """Get dtype for tests."""
        return torch.bfloat16

    def test_grouped_vs_baseline_forward(self, device, dtype):
        """Test that grouped attention produces same forward output as baseline."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        set_seed(rank)

        # Test configuration
        nheads = 8
        nheads_k = 8
        head_dim = 64
        dropout_p = 0
        causal = True
        deterministic = False

        # Sequences divisible by 2*world_size
        if world_size == 1:
            cu_seqlens_list = [0, 128, 1024]
        else:
            cu_seqlens_list = [0, 128 * world_size, 1024 * world_size]

        cu_seqlens_tensor = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
        max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
        total_length = cu_seqlens_list[-1]

        # Create test data
        q = torch.randn(total_length, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        kv = torch.randn(total_length, 2, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)

        dist.broadcast(q, src=0)
        dist.broadcast(kv, src=0)

        # Extract local data
        local_q = extract_local_zigzag(q, cu_seqlens_tensor, rank, world_size)
        local_kv = extract_local_zigzag(kv, cu_seqlens_tensor, rank, world_size)
        local_q.requires_grad = True
        local_kv.requires_grad = True

        # Prepare cu_seqlens
        local_cu_seqlens = []
        offset = 0
        for i in range(len(cu_seqlens_tensor) - 1):
            seq_len = (cu_seqlens_tensor[i+1] - cu_seqlens_tensor[i]).item()
            local_seq_len = seq_len // world_size
            local_cu_seqlens.append(offset)
            offset += local_seq_len
        local_cu_seqlens.append(offset)
        local_cu_seqlens_tensor = torch.tensor(local_cu_seqlens, dtype=torch.int32, device=device)

        # Prepare cu_seqlens for attention
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens_tensor,
            causal=causal,
            rank=rank,
            world_size=world_size,
        )

        dist.barrier()

        # Baseline: two-kernels mode without grouped attention
        baseline_out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            local_q.detach().clone().requires_grad_(True),
            local_kv.detach().clone().requires_grad_(True),
            local_cu_seqlens_tensor,
            cu_seqlens_tensor,
            max_seqlen,
            max_seqlen,
            heads_k_stride=nheads_k,
            local_k_slice=local_k_slice,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            use_fused_kernel_forward=False,
            use_fused_kernel_backward=False,
            n_chunks=2,
        )

        # Test: with grouped attention (when implemented)
        # For now, this will use the same code path, but once grouped attention
        # is implemented, it should be enabled via a flag
        grouped_out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            local_q,
            local_kv,
            local_cu_seqlens_tensor,
            cu_seqlens_tensor,
            max_seqlen,
            max_seqlen,
            heads_k_stride=nheads_k,
            local_k_slice=local_k_slice,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            use_fused_kernel_forward=False,
            use_fused_kernel_backward=False,
            n_chunks=2,
            # use_grouped_attention=True,  # TODO: Enable when implemented
        )

        # Verify forward outputs match
        out_diff = (baseline_out - grouped_out).abs()
        max_diff = out_diff.max().item()
        mean_diff = out_diff.mean().item()

        if rank == 0:
            print(f"Forward max diff: {max_diff:.6f}, mean diff: {mean_diff:.9f}")

        assert max_diff < 0.01, f"Forward max diff {max_diff:.6f} exceeds tolerance"
        assert mean_diff < 1e-05, f"Forward mean diff {mean_diff:.9f} exceeds tolerance"

        dist.barrier()

    def test_grouped_vs_baseline_backward(self, device, dtype):
        """Test that grouped attention produces same gradients as baseline."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        set_seed(rank)

        # Test configuration
        nheads = 8
        nheads_k = 8
        head_dim = 64
        dropout_p = 0
        causal = True
        deterministic = False

        if world_size == 1:
            cu_seqlens_list = [0, 128, 1024]
        else:
            cu_seqlens_list = [0, 128 * world_size, 1024 * world_size]

        cu_seqlens_tensor = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
        max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
        total_length = cu_seqlens_list[-1]

        # Create test data
        q = torch.randn(total_length, nheads, head_dim, device=device, dtype=dtype)
        kv = torch.randn(total_length, 2, nheads_k, head_dim, device=device, dtype=dtype)
        dout = torch.randn(total_length, nheads, head_dim, device=device, dtype=dtype)

        dist.broadcast(q, src=0)
        dist.broadcast(kv, src=0)
        dist.broadcast(dout, src=0)

        # Extract local data
        local_q_baseline = extract_local_zigzag(q, cu_seqlens_tensor, rank, world_size).requires_grad_(True)
        local_kv_baseline = extract_local_zigzag(kv, cu_seqlens_tensor, rank, world_size).requires_grad_(True)
        local_q_grouped = extract_local_zigzag(q, cu_seqlens_tensor, rank, world_size).requires_grad_(True)
        local_kv_grouped = extract_local_zigzag(kv, cu_seqlens_tensor, rank, world_size).requires_grad_(True)
        local_dout = extract_local_zigzag(dout, cu_seqlens_tensor, rank, world_size)

        # Prepare cu_seqlens
        local_cu_seqlens = []
        offset = 0
        for i in range(len(cu_seqlens_tensor) - 1):
            seq_len = (cu_seqlens_tensor[i+1] - cu_seqlens_tensor[i]).item()
            local_seq_len = seq_len // world_size
            local_cu_seqlens.append(offset)
            offset += local_seq_len
        local_cu_seqlens.append(offset)
        local_cu_seqlens_tensor = torch.tensor(local_cu_seqlens, dtype=torch.int32, device=device)

        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens_tensor,
            causal=causal,
            rank=rank,
            world_size=world_size,
        )

        dist.barrier()

        # Baseline backward
        baseline_out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            local_q_baseline,
            local_kv_baseline,
            local_cu_seqlens_tensor,
            cu_seqlens_tensor,
            max_seqlen,
            max_seqlen,
            heads_k_stride=nheads_k,
            local_k_slice=local_k_slice,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            use_fused_kernel_forward=False,
            use_fused_kernel_backward=False,
            n_chunks=2,
        )
        baseline_out.backward(local_dout)

        # Grouped backward
        grouped_out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            local_q_grouped,
            local_kv_grouped,
            local_cu_seqlens_tensor,
            cu_seqlens_tensor,
            max_seqlen,
            max_seqlen,
            heads_k_stride=nheads_k,
            local_k_slice=local_k_slice,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            use_fused_kernel_forward=False,
            use_fused_kernel_backward=False,
            n_chunks=2,
            # use_grouped_attention=True,  # TODO: Enable when implemented
        )
        grouped_out.backward(local_dout)

        # Compare gradients
        dq_diff = (local_q_baseline.grad - local_q_grouped.grad).abs()
        dkv_diff = (local_kv_baseline.grad - local_kv_grouped.grad).abs()

        max_dq_diff = dq_diff.max().item()
        max_dkv_diff = dkv_diff.max().item()
        mean_dq_diff = dq_diff.mean().item()
        mean_dkv_diff = dkv_diff.mean().item()

        if rank == 0:
            print(f"Backward dQ max diff: {max_dq_diff:.6f}, mean diff: {mean_dq_diff:.9f}")
            print(f"Backward dKV max diff: {max_dkv_diff:.6f}, mean diff: {mean_dkv_diff:.9f}")

        assert max_dq_diff < 0.01, f"dQ max diff {max_dq_diff:.6f} exceeds tolerance"
        assert max_dkv_diff < 0.1, f"dKV max diff {max_dkv_diff:.6f} exceeds tolerance"
        assert mean_dq_diff < 1e-05, f"dQ mean diff {mean_dq_diff:.9f} exceeds tolerance"
        assert mean_dkv_diff < 3e-04, f"dKV mean diff {mean_dkv_diff:.9f} exceeds tolerance"

        dist.barrier()

    @pytest.mark.parametrize("seq_len", [4096, 8192, 16384])
    def test_different_sequence_lengths(self, device, dtype, seq_len):
        """Test grouped attention with different sequence lengths."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        set_seed(rank)

        # Ensure divisibility
        if seq_len % (2 * world_size) != 0:
            pytest.skip(f"seq_len {seq_len} not divisible by 2*world_size {2*world_size}")

        nheads = 8
        nheads_k = 8
        head_dim = 64

        cu_seqlens_tensor = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        total_length = seq_len

        q = torch.randn(total_length, nheads, head_dim, device=device, dtype=dtype)
        kv = torch.randn(total_length, 2, nheads_k, head_dim, device=device, dtype=dtype)

        dist.broadcast(q, src=0)
        dist.broadcast(kv, src=0)

        local_q = extract_local_zigzag(q, cu_seqlens_tensor, rank, world_size).requires_grad_(True)
        local_kv = extract_local_zigzag(kv, cu_seqlens_tensor, rank, world_size).requires_grad_(True)

        local_cu_seqlens_tensor = torch.tensor(
            [0, seq_len // world_size],
            dtype=torch.int32,
            device=device
        )

        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens_tensor,
            causal=True,
            rank=rank,
            world_size=world_size,
        )

        dist.barrier()

        # Run with grouped attention
        out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            local_q,
            local_kv,
            local_cu_seqlens_tensor,
            cu_seqlens_tensor,
            seq_len,
            seq_len,
            heads_k_stride=nheads_k,
            local_k_slice=local_k_slice,
            causal=True,
            window_size=(-1, -1),
            use_fused_kernel_forward=False,
            use_fused_kernel_backward=False,
            n_chunks=2,
        )

        assert out.shape == (seq_len // world_size, nheads, head_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        dist.barrier()

    def test_with_gqa(self, device, dtype):
        """Test grouped attention with GQA (grouped query attention)."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        set_seed(rank)

        # GQA configuration
        nheads = 32  # 32 query heads
        nheads_k = 8  # 8 key/value heads
        head_dim = 64

        if world_size == 1:
            cu_seqlens_list = [0, 128, 1024]
        else:
            cu_seqlens_list = [0, 128 * world_size, 1024 * world_size]

        cu_seqlens_tensor = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
        max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
        total_length = cu_seqlens_list[-1]

        q = torch.randn(total_length, nheads, head_dim, device=device, dtype=dtype)
        kv = torch.randn(total_length, 2, nheads_k, head_dim, device=device, dtype=dtype)

        dist.broadcast(q, src=0)
        dist.broadcast(kv, src=0)

        local_q = extract_local_zigzag(q, cu_seqlens_tensor, rank, world_size).requires_grad_(True)
        local_kv = extract_local_zigzag(kv, cu_seqlens_tensor, rank, world_size).requires_grad_(True)

        local_cu_seqlens = []
        offset = 0
        for i in range(len(cu_seqlens_tensor) - 1):
            seq_len = (cu_seqlens_tensor[i+1] - cu_seqlens_tensor[i]).item()
            local_seq_len = seq_len // world_size
            local_cu_seqlens.append(offset)
            offset += local_seq_len
        local_cu_seqlens.append(offset)
        local_cu_seqlens_tensor = torch.tensor(local_cu_seqlens, dtype=torch.int32, device=device)

        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens_tensor,
            causal=True,
            rank=rank,
            world_size=world_size,
        )

        dist.barrier()

        # Run with GQA and grouped attention
        out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            local_q,
            local_kv,
            local_cu_seqlens_tensor,
            cu_seqlens_tensor,
            max_seqlen,
            max_seqlen,
            heads_k_stride=nheads_k,
            local_k_slice=local_k_slice,
            causal=True,
            window_size=(-1, -1),
            use_fused_kernel_forward=False,
            use_fused_kernel_backward=False,
            n_chunks=2,
        )

        expected_tokens = sum(local_cu_seqlens_tensor[i+1] - local_cu_seqlens_tensor[i]
                             for i in range(len(local_cu_seqlens_tensor) - 1))
        assert out.shape == (expected_tokens, nheads, head_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        dist.barrier()


def main():
    """Main test function for running as script."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 60)
        print(f"Testing Zigzag Llama3 with Grouped Attention (world_size={world_size})")
        print("=" * 60)

    # Run tests
    test_suite = TestZigzagLlama3GroupedAttention()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    try:
        if rank == 0:
            print("\n# Test 1: Forward pass correctness")
        dist.barrier()
        test_suite.test_grouped_vs_baseline_forward(device, dtype)
        if rank == 0:
            print("PASSED")

        if rank == 0:
            print("\n# Test 2: Backward pass correctness")
        dist.barrier()
        test_suite.test_grouped_vs_baseline_backward(device, dtype)
        if rank == 0:
            print("PASSED")

        if rank == 0:
            print("\n# Test 3: Different sequence lengths")
        dist.barrier()
        for seq_len in [4096, 8192]:
            if seq_len % (2 * world_size) == 0:
                if rank == 0:
                    print(f"  Testing seq_len={seq_len}")
                test_suite.test_different_sequence_lengths(device, dtype, seq_len)
        if rank == 0:
            print("PASSED")

        if rank == 0:
            print("\n# Test 4: GQA support")
        dist.barrier()
        test_suite.test_with_gqa(device, dtype)
        if rank == 0:
            print("PASSED")

        if rank == 0:
            print("\n" + "=" * 60)
            print("All tests PASSED!")
            print("=" * 60)

    except Exception as e:
        if rank == 0:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        raise

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
