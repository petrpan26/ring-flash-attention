#!/usr/bin/env python3
"""
Example usage of grouped flash attention with zigzag_llama3.

This demonstrates how to use the new use_grouped_attention parameter
to enable the Python prototype grouped attention (Option B).
"""

import torch
import torch.distributed as dist
from ring_flash_attn import zigzag_llama3_flash_attn_varlen_func


def example_grouped_attention():
    """
    Example showing how to use grouped attention in zigzag_llama3.

    This example demonstrates the three execution modes:
    1. Two-kernels mode (baseline)
    2. Grouped attention mode (Option B - Python prototype)
    3. Triton grouped mode (Option C - custom Triton kernel)
    """

    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')

    # Example parameters (adjust for your use case)
    batch_size = 4
    seqlen_per_rank = 1024  # Local sequence length per rank
    nheads = 32
    nheads_k = 8  # GQA
    head_dim = 128
    heads_k_stride = 4

    # Create input tensors (in zigzag interleaved format)
    q = torch.randn(batch_size * seqlen_per_rank, nheads, head_dim,
                    device=device, dtype=torch.float16)
    k = torch.randn(batch_size * seqlen_per_rank, nheads_k, head_dim,
                    device=device, dtype=torch.float16)
    v = torch.randn(batch_size * seqlen_per_rank, nheads_k, head_dim,
                    device=device, dtype=torch.float16)

    # Create cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_per_rank,
                                 seqlen_per_rank, device=device, dtype=torch.int32)
    # Global cu_seqlens_k (total across all ranks)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_per_rank * world_size,
                                 seqlen_per_rank * world_size, device=device, dtype=torch.int32)

    max_seqlen_q = seqlen_per_rank
    max_seqlen_k = seqlen_per_rank * world_size
    local_k_slice = slice(None)  # Use all local K,V

    softmax_scale = 1.0 / (head_dim ** 0.5)

    print(f"[Rank {rank}] Running attention with different modes...")

    # Mode 1: Two-kernels mode (baseline)
    print(f"[Rank {rank}] Mode 1: Two-kernels (baseline)")
    out_baseline = zigzag_llama3_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        heads_k_stride, local_k_slice,
        softmax_scale=softmax_scale,
        causal=True,
        use_fused_kernel_forward=False,
        n_chunks=2,
        use_triton_grouped=False,
        use_grouped_attention=False,  # Baseline mode
    )

    # Mode 2: Grouped attention (Option B - Python prototype)
    print(f"[Rank {rank}] Mode 2: Grouped attention (Python prototype)")
    out_grouped = zigzag_llama3_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        heads_k_stride, local_k_slice,
        softmax_scale=softmax_scale,
        causal=True,
        use_fused_kernel_forward=False,
        n_chunks=2,
        use_triton_grouped=False,
        use_grouped_attention=True,  # Enable grouped attention!
    )

    # Verify outputs match
    if torch.allclose(out_baseline, out_grouped, rtol=1e-3, atol=1e-3):
        print(f"[Rank {rank}] SUCCESS: Grouped attention matches baseline!")
    else:
        max_diff = (out_baseline - out_grouped).abs().max().item()
        print(f"[Rank {rank}] WARNING: Outputs differ by max {max_diff}")

    # Mode 3: Triton grouped mode (Option C - if available)
    # This requires the custom Triton kernel implementation
    try:
        print(f"[Rank {rank}] Mode 3: Triton grouped mode")
        out_triton = zigzag_llama3_flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            heads_k_stride, local_k_slice,
            softmax_scale=softmax_scale,
            causal=True,
            use_fused_kernel_forward=False,
            n_chunks=2,
            use_triton_grouped=True,  # Enable Triton grouped kernel!
            use_grouped_attention=False,
        )

        if torch.allclose(out_baseline, out_triton, rtol=1e-3, atol=1e-3):
            print(f"[Rank {rank}] SUCCESS: Triton grouped matches baseline!")
        else:
            max_diff = (out_baseline - out_triton).abs().max().item()
            print(f"[Rank {rank}] WARNING: Outputs differ by max {max_diff}")
    except Exception as e:
        print(f"[Rank {rank}] Triton grouped mode not available: {e}")

    print(f"[Rank {rank}] All modes completed!")


if __name__ == "__main__":
    example_grouped_attention()
