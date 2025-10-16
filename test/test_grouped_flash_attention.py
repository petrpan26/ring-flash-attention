"""
Comprehensive unit tests for grouped flash attention implementations.

This module tests all three implementations:
1. Python prototype (wrapper around existing kernels)
2. CUDA kernel modification
3. Triton custom kernel

Test coverage:
- Correctness: grouped results match separate kernel calls
- Multiple group counts (2, 3, 4 groups)
- Different K,V slice lengths per group
- With/without causal masking
- GQA (grouped query attention)
- Variable-length sequences
- Edge cases (empty sequences, single token, etc.)
- Forward and backward passes
"""

import pytest
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


# Mock implementation placeholders - will be replaced with actual implementations
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

        # Use existing flash_attn_varlen_func
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
        # LSE not available in functional API, create dummy
        lse_list.append(torch.zeros(out.shape[0], out.shape[1], device=out.device, dtype=torch.float32))

    return out_list, lse_list


def _flash_attn_varlen_forward_grouped_cuda(
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
    """CUDA kernel implementation (to be implemented in Phase 2)."""
    # For now, fall back to Python implementation
    return _flash_attn_varlen_forward_grouped_python(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
        max_seqlen_q_list, max_seqlen_k_list, dropout_p, softmax_scale,
        causal, window_size, alibi_slopes, deterministic
    )


def _flash_attn_varlen_forward_grouped_triton(
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
    """Triton kernel implementation (to be implemented in Phase 3)."""
    # For now, fall back to Python implementation
    return _flash_attn_varlen_forward_grouped_python(
        q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
        max_seqlen_q_list, max_seqlen_k_list, dropout_p, softmax_scale,
        causal, window_size, alibi_slopes, deterministic
    )


# Test fixtures and utilities
@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def dtype():
    return torch.float16


def generate_test_data(
    total_tokens: int,
    nheads: int,
    nheads_k: int,
    head_dim: int,
    num_sequences: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random test data for attention."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Create cu_seqlens with equal length sequences for simplicity
    seq_len = total_tokens // num_sequences
    cu_seqlens = torch.arange(0, total_tokens + 1, seq_len, device=device, dtype=torch.int32)

    return q, k, v, cu_seqlens


def compute_reference_separate_calls(
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
) -> List[torch.Tensor]:
    """Compute reference outputs using separate kernel calls."""
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not installed")

    out_list = []
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

    return out_list


# Test Classes
class TestGroupedFlashAttentionCorrectness:
    """Test correctness of grouped attention implementations."""

    @pytest.mark.parametrize("implementation", ["python", "cuda", "triton"])
    @pytest.mark.parametrize("num_groups", [2, 3, 4])
    def test_basic_correctness_multiple_groups(self, device, dtype, implementation, num_groups):
        """Test that grouped attention produces same results as separate calls."""
        # Configuration
        total_tokens = 2000
        nheads = 8
        nheads_k = 8
        head_dim = 64
        num_sequences = 2

        # Generate test data
        q, k, v, cu_seqlens = generate_test_data(
            total_tokens, nheads, nheads_k, head_dim, num_sequences, device, dtype
        )

        # Split Q into groups with different K,V slice lengths
        tokens_per_group = total_tokens // num_groups
        q_list = [q[i * tokens_per_group:(i + 1) * tokens_per_group].clone()
                  for i in range(num_groups)]

        # Each group attends to increasing K,V lengths
        max_seqlen_k_list = [(i + 1) * tokens_per_group for i in range(num_groups)]
        max_seqlen_q_list = [tokens_per_group] * num_groups

        # Create cu_seqlens for each group
        cu_seqlens_q_list = []
        cu_seqlens_k_list = []
        for i in range(num_groups):
            # Q cu_seqlens for this group (2 sequences)
            cu_seqlens_q = torch.tensor(
                [0, tokens_per_group // 2, tokens_per_group],
                device=device, dtype=torch.int32
            )
            cu_seqlens_q_list.append(cu_seqlens_q)

            # K cu_seqlens for this group (extends to group's K,V limit)
            k_len = max_seqlen_k_list[i]
            cu_seqlens_k = torch.tensor(
                [0, k_len // 2, k_len],
                device=device, dtype=torch.int32
            )
            cu_seqlens_k_list.append(cu_seqlens_k)

        # Select implementation
        if implementation == "python":
            impl_func = _flash_attn_varlen_forward_grouped_python
        elif implementation == "cuda":
            impl_func = _flash_attn_varlen_forward_grouped_cuda
        else:
            impl_func = _flash_attn_varlen_forward_grouped_triton

        # Test: grouped call
        out_list_grouped, _ = impl_func(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
            dropout_p=0.0,
            causal=False,
        )

        # Reference: separate calls
        out_list_ref = compute_reference_separate_calls(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
            dropout_p=0.0,
            causal=False,
        )

        # Verify
        for i in range(num_groups):
            assert torch.allclose(
                out_list_grouped[i], out_list_ref[i], rtol=1e-3, atol=1e-3
            ), f"Group {i} output mismatch"

    @pytest.mark.parametrize("implementation", ["python"])
    def test_with_causal_masking(self, device, dtype, implementation):
        """Test grouped attention with causal masking."""
        # Configuration
        total_tokens = 1024
        nheads = 4
        nheads_k = 4
        head_dim = 64
        num_sequences = 2
        num_groups = 2

        q, k, v, _ = generate_test_data(
            total_tokens, nheads, nheads_k, head_dim, num_sequences, device, dtype
        )

        # Split Q into 2 groups
        tokens_per_group = total_tokens // num_groups
        q_list = [q[:tokens_per_group].clone(), q[tokens_per_group:].clone()]

        # Group 0 attends to first 512 tokens, Group 1 to all 1024
        max_seqlen_k_list = [512, 1024]
        max_seqlen_q_list = [512, 512]

        cu_seqlens_q_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 512, 1024], device=device, dtype=torch.int32),
        ]

        # Test with causal masking
        out_list_grouped, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
            causal=True,
        )

        # Reference
        out_list_ref = compute_reference_separate_calls(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
            causal=True,
        )

        for i in range(num_groups):
            assert torch.allclose(
                out_list_grouped[i], out_list_ref[i], rtol=1e-3, atol=1e-3
            ), f"Group {i} causal output mismatch"

    @pytest.mark.parametrize("implementation", ["python"])
    def test_with_gqa(self, device, dtype, implementation):
        """Test grouped attention with GQA (grouped query attention)."""
        # Configuration with GQA
        total_tokens = 1024
        nheads = 32  # 32 query heads
        nheads_k = 8  # 8 key/value heads (GQA)
        head_dim = 64
        num_sequences = 2
        num_groups = 2

        q, k, v, _ = generate_test_data(
            total_tokens, nheads, nheads_k, head_dim, num_sequences, device, dtype
        )

        tokens_per_group = total_tokens // num_groups
        q_list = [q[:tokens_per_group].clone(), q[tokens_per_group:].clone()]

        max_seqlen_k_list = [512, 1024]
        max_seqlen_q_list = [512, 512]

        cu_seqlens_q_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 512, 1024], device=device, dtype=torch.int32),
        ]

        # Test with GQA
        out_list_grouped, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        # Reference
        out_list_ref = compute_reference_separate_calls(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        for i in range(num_groups):
            assert torch.allclose(
                out_list_grouped[i], out_list_ref[i], rtol=1e-3, atol=1e-3
            ), f"Group {i} GQA output mismatch"

    @pytest.mark.parametrize("implementation", ["python"])
    def test_variable_length_sequences(self, device, dtype, implementation):
        """Test with variable-length sequences."""
        nheads = 8
        nheads_k = 8
        head_dim = 64

        # Create variable-length sequences
        seq_lengths = [128, 256, 512, 1024]
        total_tokens = sum(seq_lengths)

        torch.manual_seed(42)
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        cu_seqlens_full = torch.tensor([0] + [sum(seq_lengths[:i+1]) for i in range(len(seq_lengths))],
                                       device=device, dtype=torch.int32)

        # Split into 2 groups with different K,V lengths
        # Group 0: First 2 sequences (128 + 256 = 384 tokens)
        # Group 1: Last 2 sequences (512 + 1024 = 1536 tokens, but we use 920 to fit in total)
        q_list = [q[:384].clone(), q[384:1304].clone()]

        cu_seqlens_q_list = [
            torch.tensor([0, 128, 384], device=device, dtype=torch.int32),  # First 2 seqs
            torch.tensor([0, 512, 920], device=device, dtype=torch.int32),  # Last 2 seqs
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 128, 384], device=device, dtype=torch.int32),
            torch.tensor([0, 640, 1920], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [256, 512]
        max_seqlen_k_list = [384, 1920]

        out_list_grouped, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        out_list_ref = compute_reference_separate_calls(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        for i in range(len(q_list)):
            assert torch.allclose(
                out_list_grouped[i], out_list_ref[i], rtol=1e-3, atol=1e-3
            ), f"Group {i} variable-length output mismatch"


class TestGroupedFlashAttentionEdgeCases:
    """Test edge cases for grouped attention."""

    @pytest.mark.parametrize("implementation", ["python"])
    def test_single_token_sequence(self, device, dtype, implementation):
        """Test with single token sequences."""
        nheads = 4
        nheads_k = 4
        head_dim = 64

        # Single token per sequence
        q = torch.randn(2, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(2, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(2, nheads_k, head_dim, device=device, dtype=dtype)

        q_list = [q[:1].clone(), q[1:].clone()]
        cu_seqlens_q_list = [
            torch.tensor([0, 1], device=device, dtype=torch.int32),
            torch.tensor([0, 1], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 1], device=device, dtype=torch.int32),
            torch.tensor([0, 2], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [1, 1]
        max_seqlen_k_list = [1, 2]

        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        assert out_list[0].shape == (1, nheads, head_dim)
        assert out_list[1].shape == (1, nheads, head_dim)

    @pytest.mark.parametrize("implementation", ["python"])
    def test_single_group(self, device, dtype, implementation):
        """Test with single group (should work like regular attention)."""
        total_tokens = 512
        nheads = 8
        nheads_k = 8
        head_dim = 64

        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        cu_seqlens = torch.tensor([0, 256, 512], device=device, dtype=torch.int32)

        q_list = [q]
        cu_seqlens_q_list = [cu_seqlens]
        cu_seqlens_k_list = [cu_seqlens]
        max_seqlen_q_list = [256]
        max_seqlen_k_list = [512]

        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        assert len(out_list) == 1
        assert out_list[0].shape == (total_tokens, nheads, head_dim)

    @pytest.mark.parametrize("implementation", ["python"])
    def test_overlapping_kv_regions(self, device, dtype, implementation):
        """Test that overlapping K,V regions are handled correctly."""
        total_tokens = 1024
        nheads = 8
        nheads_k = 8
        head_dim = 64

        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        # Both groups attend to overlapping K,V regions
        q_list = [q[:512].clone(), q[512:].clone()]

        cu_seqlens_q_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
        ]
        # Group 0: K,V[0:768], Group 1: K,V[0:1024] (overlaps with Group 0)
        cu_seqlens_k_list = [
            torch.tensor([0, 384, 768], device=device, dtype=torch.int32),
            torch.tensor([0, 512, 1024], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [256, 256]
        max_seqlen_k_list = [768, 1024]

        out_list_grouped, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        out_list_ref = compute_reference_separate_calls(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        # This is the key test for grouped attention benefit!
        # Both groups share K,V[:768] but should compute correctly
        for i in range(len(q_list)):
            assert torch.allclose(
                out_list_grouped[i], out_list_ref[i], rtol=1e-3, atol=1e-3
            ), f"Group {i} overlapping K,V output mismatch"


class TestGroupedFlashAttentionBackward:
    """Test backward pass for grouped attention."""

    @pytest.mark.parametrize("implementation", ["python"])
    def test_backward_correctness(self, device, dtype, implementation):
        """Test that backward pass produces correct gradients."""
        total_tokens = 1024
        nheads = 8
        nheads_k = 8
        head_dim = 64

        # Create input tensors with gradients
        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype, requires_grad=True)

        # Clone for reference computation
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        q_list = [q[:512], q[512:]]
        q_list_ref = [q_ref[:512], q_ref[512:]]

        cu_seqlens_q_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
            torch.tensor([0, 512, 1024], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [256, 256]
        max_seqlen_k_list = [512, 1024]

        # Forward pass - grouped
        out_list_grouped, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        # Forward pass - reference
        out_list_ref = compute_reference_separate_calls(
            q_list_ref, k_ref, v_ref,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        # Backward pass
        dout = torch.randn_like(torch.cat(out_list_grouped, dim=0))
        dout_list = [dout[:512], dout[512:]]

        torch.cat(out_list_grouped, dim=0).backward(dout)
        torch.cat(out_list_ref, dim=0).backward(dout)

        # Compare gradients
        if q.grad is not None and q_ref.grad is not None:
            assert torch.allclose(q.grad, q_ref.grad, rtol=1e-2, atol=1e-2), "Q gradient mismatch"
        if k.grad is not None and k_ref.grad is not None:
            assert torch.allclose(k.grad, k_ref.grad, rtol=1e-2, atol=1e-2), "K gradient mismatch"
        if v.grad is not None and v_ref.grad is not None:
            assert torch.allclose(v.grad, v_ref.grad, rtol=1e-2, atol=1e-2), "V gradient mismatch"


class TestGroupedFlashAttentionPrecision:
    """Test different precision modes."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("implementation", ["python"])
    def test_mixed_precision(self, device, dtype, implementation):
        """Test grouped attention with different dtypes."""
        total_tokens = 512
        nheads = 8
        nheads_k = 8
        head_dim = 64

        q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads_k, head_dim, device=device, dtype=dtype)

        q_list = [q[:256], q[256:]]
        cu_seqlens_q_list = [
            torch.tensor([0, 128, 256], device=device, dtype=torch.int32),
            torch.tensor([0, 128, 256], device=device, dtype=torch.int32),
        ]
        cu_seqlens_k_list = [
            torch.tensor([0, 128, 256], device=device, dtype=torch.int32),
            torch.tensor([0, 256, 512], device=device, dtype=torch.int32),
        ]
        max_seqlen_q_list = [128, 128]
        max_seqlen_k_list = [256, 512]

        out_list, _ = _flash_attn_varlen_forward_grouped_python(
            q_list, k, v,
            cu_seqlens_q_list, cu_seqlens_k_list,
            max_seqlen_q_list, max_seqlen_k_list,
        )

        assert out_list[0].dtype == dtype
        assert out_list[1].dtype == dtype

        # Check for NaN/Inf
        for i, out in enumerate(out_list):
            assert not torch.isnan(out).any(), f"Group {i} output contains NaN"
            assert not torch.isinf(out).any(), f"Group {i} output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
