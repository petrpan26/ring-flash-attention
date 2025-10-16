"""
Stress tests for grouped flash attention implementations.

This module tests:
- Large batch sizes
- Long sequences (128K+ tokens)
- Many groups (8+)
- Mixed precision (fp16, bf16)
- Memory pressure scenarios
- Edge cases under stress

Run with:
    pytest test/test_grouped_flash_attention_stress.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import gc


# Mock implementation (same as in unit tests)
def _flash_attn_varlen_forward_grouped_python(
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
    deterministic: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Python prototype using existing flash_attn kernels."""
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not installed")

    out_list = []
    lse_list = []

    for i, q_group in enumerate(q_list):
        k_end = cu_seqlens_k_list[i][-1].item()

        out = flash_attn_varlen_func(
            q=q_group,
            k=k[:k_end],
            v=v[:k_end],
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=max_seqlen_q_list[i],
            max_seqlen_k=max_seqlen_k_list[i],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )

        out_list.append(out)
        lse_list.append(torch.zeros(out.shape[0], out.shape[1], device=out.device, dtype=torch.float32))

    return out_list, lse_list


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup GPU memory between tests."""
    yield
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class TestGroupedFlashAttentionLargeScales:
    """Test grouped attention with large-scale inputs."""

    @pytest.mark.parametrize("seq_len", [65536, 131072])
    def test_long_sequences(self, device, seq_len):
        """Test with very long sequences (64K-128K tokens)."""
        dtype = torch.bfloat16
        nheads = 32
        nheads_k = 8
        head_dim = 128
        num_groups = 2

        # Check if we have enough memory
        estimated_memory_gb = (seq_len * nheads * head_dim * 2 * 3) / (1024 ** 3)  # Q, K, V
        available_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

        if estimated_memory_gb * 2 > available_memory_gb:
            pytest.skip(f"Not enough GPU memory: need ~{estimated_memory_gb*2:.1f}GB, have {available_memory_gb:.1f}GB")

        torch.manual_seed(42)
        q = torch.randn(seq_len, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(seq_len, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(seq_len, nheads_k, head_dim, device=device, dtype=dtype)

        # Split into groups
        tokens_per_group = seq_len // num_groups
        q_list = [q[i * tokens_per_group:(i + 1) * tokens_per_group].clone()
                  for i in range(num_groups)]

        max_seqlen_k_list = [(i + 1) * tokens_per_group for i in range(num_groups)]
        max_seqlen_q_list = [tokens_per_group] * num_groups

        cu_seqlens_q_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32)
            for _ in range(num_groups)
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, max_seqlen_k_list[i]], device=device, dtype=torch.int32)
            for i in range(num_groups)
        ]

        # Run grouped attention
        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
            causal=True,
        )

        # Verify outputs
        for i, out in enumerate(out_list):
            assert out.shape == (tokens_per_group, nheads, head_dim)
            assert not torch.isnan(out).any(), f"Group {i} contains NaN"
            assert not torch.isinf(out).any(), f"Group {i} contains Inf"

    @pytest.mark.parametrize("num_groups", [8, 16])
    def test_many_groups(self, device, num_groups):
        """Test with many groups (8-16 groups)."""
        dtype = torch.bfloat16
        total_tokens = 16384
        nheads = 32
        nheads_k = 8
        head_dim = 128

        torch.manual_seed(42)
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        # Split into many groups
        tokens_per_group = total_tokens // num_groups
        q_list = [q[i * tokens_per_group:(i + 1) * tokens_per_group].clone()
                  for i in range(num_groups)]

        max_seqlen_k_list = [(i + 1) * tokens_per_group for i in range(num_groups)]
        max_seqlen_q_list = [tokens_per_group] * num_groups

        cu_seqlens_q_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32)
            for _ in range(num_groups)
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, max_seqlen_k_list[i]], device=device, dtype=torch.int32)
            for i in range(num_groups)
        ]

        # Run grouped attention
        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        assert len(out_list) == num_groups
        for i, out in enumerate(out_list):
            assert out.shape == (tokens_per_group, nheads, head_dim)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_large_batch_multiple_sequences(self, device):
        """Test with large batch (many sequences)."""
        dtype = torch.bfloat16
        nheads = 32
        nheads_k = 8
        head_dim = 128
        num_sequences = 32  # Large batch
        seq_len_per_batch = 512
        total_tokens = num_sequences * seq_len_per_batch

        torch.manual_seed(42)
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        # Create cu_seqlens for multiple sequences
        cu_seqlens_full = torch.arange(
            0, total_tokens + 1, seq_len_per_batch,
            device=device, dtype=torch.int32
        )

        # Split into 2 groups
        num_groups = 2
        tokens_per_group = total_tokens // num_groups
        q_list = [q[:tokens_per_group].clone(), q[tokens_per_group:].clone()]

        # Each group has half the sequences
        seqs_per_group = num_sequences // num_groups
        cu_seqlens_q_list = [
            torch.arange(0, tokens_per_group + 1, seq_len_per_batch,
                        device=device, dtype=torch.int32),
            torch.arange(0, tokens_per_group + 1, seq_len_per_batch,
                        device=device, dtype=torch.int32),
        ]

        cu_seqlens_k_list = [
            torch.arange(0, tokens_per_group + 1, seq_len_per_batch,
                        device=device, dtype=torch.int32),
            torch.arange(0, total_tokens + 1, seq_len_per_batch * 2,
                        device=device, dtype=torch.int32),
        ]

        max_seqlen_q_list = [seq_len_per_batch, seq_len_per_batch]
        max_seqlen_k_list = [seq_len_per_batch, seq_len_per_batch * 2]

        # Run grouped attention
        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        for i, out in enumerate(out_list):
            assert out.shape == (tokens_per_group, nheads, head_dim)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


class TestGroupedFlashAttentionMemoryPressure:
    """Test grouped attention under memory pressure."""

    def test_backward_pass_large_scale(self, device):
        """Test backward pass with large tensors."""
        dtype = torch.bfloat16
        total_tokens = 32768
        nheads = 32
        nheads_k = 8
        head_dim = 128
        num_groups = 2

        # Check memory
        estimated_memory_gb = (total_tokens * nheads * head_dim * 2 * 6) / (1024 ** 3)
        available_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

        if estimated_memory_gb * 2 > available_memory_gb:
            pytest.skip(f"Not enough GPU memory for backward pass")

        torch.manual_seed(42)
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)

        tokens_per_group = total_tokens // num_groups
        q_list = [q[:tokens_per_group], q[tokens_per_group:]]

        cu_seqlens_q_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
            torch.tensor([0, total_tokens], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [tokens_per_group, tokens_per_group]
        max_seqlen_k_list = [tokens_per_group, total_tokens]

        # Forward
        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        # Backward
        dout = torch.randn_like(torch.cat(out_list, dim=0))
        torch.cat(out_list, dim=0).backward(dout)

        # Verify gradients exist and are valid
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_repeated_calls_no_memory_leak(self, device):
        """Test that repeated calls don't leak memory."""
        dtype = torch.bfloat16
        total_tokens = 8192
        nheads = 32
        nheads_k = 8
        head_dim = 128
        num_groups = 2

        # Get baseline memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated(device)

        for iteration in range(10):
            torch.manual_seed(42)
            q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
            k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
            v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

            tokens_per_group = total_tokens // num_groups
            q_list = [q[:tokens_per_group].clone(), q[tokens_per_group:].clone()]

            cu_seqlens_q_list = [
                torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
                torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
            ]
            cu_seqlens_k_list = [
                torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
                torch.tensor([0, total_tokens], device=device, dtype=torch.int32),
            ]
            max_seqlen_q_list = [tokens_per_group, tokens_per_group]
            max_seqlen_k_list = [tokens_per_group, total_tokens]

            out_list, _ = _flash_attn_varlen_forward_grouped_python(
                q_list, k, v,
                cu_seqlens_q_list, cu_seqlens_k_list,
                max_seqlen_q_list, max_seqlen_k_list,
            )

            # Clean up
            del q, k, v, q_list, out_list
            gc.collect()

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated(device)
        memory_increase = final_memory - baseline_memory

        # Allow small increase (due to caching), but not proportional to iterations
        max_allowed_increase = 100 * 1024 * 1024  # 100MB
        assert memory_increase < max_allowed_increase, \
            f"Memory leak detected: {memory_increase / (1024**2):.1f}MB increase after 10 iterations"


class TestGroupedFlashAttentionPrecisionStress:
    """Test precision handling under stress."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_different_dtypes_large_scale(self, device, dtype):
        """Test different dtypes with large tensors."""
        total_tokens = 16384
        nheads = 32
        nheads_k = 8
        head_dim = 128
        num_groups = 2

        torch.manual_seed(42)
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        tokens_per_group = total_tokens // num_groups
        q_list = [q[:tokens_per_group].clone(), q[tokens_per_group:].clone()]

        cu_seqlens_q_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32),
            torch.tensor([0, total_tokens], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [tokens_per_group, tokens_per_group]
        max_seqlen_k_list = [tokens_per_group, total_tokens]

        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        for i, out in enumerate(out_list):
            assert out.dtype == dtype
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
            # Check reasonable value range
            assert out.abs().max() < 100.0, f"Group {i} has unreasonably large values"

    def test_extreme_values_handling(self, device):
        """Test handling of extreme input values."""
        dtype = torch.bfloat16
        total_tokens = 1024
        nheads = 8
        nheads_k = 8
        head_dim = 64

        # Create inputs with extreme values
        torch.manual_seed(42)
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype) * 10
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype) * 10
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype) * 10

        q_list = [q[:512].clone(), q[512:].clone()]
        cu_seqlens_q_list = [
            torch.tensor([0, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 512], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 1024], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [512, 512]
        max_seqlen_k_list = [512, 1024]

        # Should handle extreme values without overflow
        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        for i, out in enumerate(out_list):
            assert not torch.isnan(out).any(), f"Group {i} produced NaN with extreme values"
            assert not torch.isinf(out).any(), f"Group {i} produced Inf with extreme values"


class TestGroupedFlashAttentionEdgeCasesStress:
    """Test edge cases under stress conditions."""

    def test_highly_imbalanced_groups(self, device):
        """Test with very imbalanced group sizes."""
        dtype = torch.bfloat16
        nheads = 8
        nheads_k = 8
        head_dim = 64

        # Very imbalanced: first group tiny, second group huge
        total_tokens = 10000
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        # Group 0: 100 tokens, Group 1: 9900 tokens
        q_list = [q[:100].clone(), q[100:].clone()]

        cu_seqlens_q_list = [
            torch.tensor([0, 100], device=device, dtype=torch.int32),
            torch.tensor([0, 9900], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 100], device=device, dtype=torch.int32),
            torch.tensor([0, 10000], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [100, 9900]
        max_seqlen_k_list = [100, 10000]

        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        assert out_list[0].shape == (100, nheads, head_dim)
        assert out_list[1].shape == (9900, nheads, head_dim)
        for i, out in enumerate(out_list):
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_all_groups_overlapping(self, device):
        """Test where all groups use same K,V region."""
        dtype = torch.bfloat16
        total_tokens = 4096
        nheads = 8
        nheads_k = 8
        head_dim = 64

        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        # All 4 groups use the same K,V
        tokens_per_group = total_tokens // 4
        q_list = [q[i * tokens_per_group:(i + 1) * tokens_per_group].clone()
                  for i in range(4)]

        cu_seqlens_q_list = [
            torch.tensor([0, tokens_per_group], device=device, dtype=torch.int32)
            for _ in range(4)
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, total_tokens], device=device, dtype=torch.int32)
            for _ in range(4)
        ]
        max_seqlen_q_list = [tokens_per_group] * 4
        max_seqlen_k_list = [total_tokens] * 4

        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        assert len(out_list) == 4
        for i, out in enumerate(out_list):
            assert out.shape == (tokens_per_group, nheads, head_dim)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
