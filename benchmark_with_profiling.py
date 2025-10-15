"""
Benchmark with GPU Profiling: Zigzag Ring vs Zigzag Llama3
Generates flamegraphs and timeline traces to identify performance bottlenecks
"""
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import os
import argparse
from functools import partial

# Import the functions to benchmark
from ring_flash_attn import (
    zigzag_ring_flash_attn_varlen_kvpacked_func,
    zigzag_llama3_flash_attn_varlen_kvpacked_func,
)


def setup_distributed():
    """Initialize distributed environment"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    return rank, world_size


def generate_test_data(rank, world_size, seqlen=2048, batch_size=2, nheads=32, d=128):
    """Generate test data in zigzag interleaved format"""
    device = torch.device(f'cuda:{rank}')

    # Calculate local sequence length per rank (zigzag interleaved)
    total_chunks = 2 * world_size
    chunk_size = seqlen // total_chunks
    local_seqlen = 2 * chunk_size  # Each rank gets 2 chunks

    # Generate Q, K, V
    q = torch.randn(
        batch_size * local_seqlen, nheads, d,
        device=device, dtype=torch.bfloat16, requires_grad=True
    )

    kv = torch.randn(
        batch_size * local_seqlen, 2, nheads, d,
        device=device, dtype=torch.bfloat16, requires_grad=True
    )

    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor(
        [i * local_seqlen for i in range(batch_size + 1)],
        device=device, dtype=torch.int32
    )

    # Global cumulative sequence lengths for K
    cu_seqlens_k = torch.tensor(
        [i * seqlen for i in range(batch_size + 1)],
        device=device, dtype=torch.int32
    )

    return q, kv, cu_seqlens_q, cu_seqlens_k


def benchmark_zigzag_ring(q, kv, cu_seqlens_q, cu_seqlens_k, rank, world_size):
    """Benchmark zigzag ring with profiling annotations"""
    with record_function("zigzag_ring_forward"):
        output = zigzag_ring_flash_attn_varlen_kvpacked_func(
            q, kv,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=q.shape[0] // len(cu_seqlens_q) - 1,
            max_seqlen_k=(cu_seqlens_k[1] - cu_seqlens_k[0]).item(),
            dropout_p=0.0,
            causal=True,
        )

    # Backward pass
    grad_output = torch.randn_like(output)
    with record_function("zigzag_ring_backward"):
        output.backward(grad_output)

    return output


def benchmark_zigzag_llama3(q, kv, cu_seqlens_q, cu_seqlens_k, rank, world_size):
    """Benchmark zigzag llama3 with profiling annotations"""
    with record_function("zigzag_llama3_forward"):
        output = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            q, kv,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=q.shape[0] // (len(cu_seqlens_q) - 1),
            max_seqlen_k=(cu_seqlens_k[1] - cu_seqlens_k[0]).item(),
            dropout_p=0.0,
            causal=True,
            use_triton_kernel=True,  # Use Triton optimization
        )

    # Backward pass
    grad_output = torch.randn_like(output)
    with record_function("zigzag_llama3_backward"):
        output.backward(grad_output)

    return output


def run_profiled_benchmark(benchmark_fn, name, q, kv, cu_seqlens_q, cu_seqlens_k, rank, world_size, warmup=3, iterations=10):
    """Run benchmark with PyTorch profiler"""

    # Warmup
    print(f"[Rank {rank}] Warming up {name}...")
    for _ in range(warmup):
        q_copy = q.detach().clone().requires_grad_(True)
        kv_copy = kv.detach().clone().requires_grad_(True)
        benchmark_fn(q_copy, kv_copy, cu_seqlens_q, cu_seqlens_k, rank, world_size)
        torch.cuda.synchronize()

    # Profiled run
    print(f"[Rank {rank}] Profiling {name}...")

    # PyTorch profiler configuration
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/{name}_rank{rank}'),
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=iterations - 3, repeat=1),
    ) as prof:
        for i in range(iterations):
            with record_function(f"iteration_{i}"):
                q_copy = q.detach().clone().requires_grad_(True)
                kv_copy = kv.detach().clone().requires_grad_(True)
                benchmark_fn(q_copy, kv_copy, cu_seqlens_q, cu_seqlens_k, rank, world_size)
                torch.cuda.synchronize()
            prof.step()

    # Export trace for visualization
    if rank == 0:
        # Export Chrome trace (for chrome://tracing)
        trace_path = f'./profiler_logs/{name}_trace_rank{rank}.json'
        prof.export_chrome_trace(trace_path)
        print(f"[Rank {rank}] Chrome trace saved to: {trace_path}")

        # Print key statistics
        print(f"\n{'='*80}")
        print(f"{name} - Key Profiling Stats (Rank {rank})")
        print(f"{'='*80}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Export stacks for flamegraph
        stacks_path = f'./profiler_logs/{name}_stacks_rank{rank}.txt'
        prof.export_stacks(stacks_path, "self_cuda_time_total")
        print(f"[Rank {rank}] Stack traces saved to: {stacks_path}")
        print(f"[Rank {rank}] Generate flamegraph with: flamegraph.pl {stacks_path} > {name}_flamegraph.svg")


def run_nsight_benchmark(benchmark_fn, name, q, kv, cu_seqlens_q, cu_seqlens_k, rank, world_size, warmup=3):
    """Run benchmark ready for Nsight Systems profiling"""

    # Warmup
    for _ in range(warmup):
        q_copy = q.detach().clone().requires_grad_(True)
        kv_copy = kv.detach().clone().requires_grad_(True)
        benchmark_fn(q_copy, kv_copy, cu_seqlens_q, cu_seqlens_k, rank, world_size)
        torch.cuda.synchronize()

    # NVTX markers for better visualization in Nsight
    torch.cuda.nvtx.range_push(f"{name}_full_iteration")

    q_copy = q.detach().clone().requires_grad_(True)
    kv_copy = kv.detach().clone().requires_grad_(True)

    torch.cuda.nvtx.range_push(f"{name}_forward")
    if "ring" in name:
        output = zigzag_ring_flash_attn_varlen_kvpacked_func(
            q_copy, kv_copy, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=q.shape[0] // (len(cu_seqlens_q) - 1),
            max_seqlen_k=(cu_seqlens_k[1] - cu_seqlens_k[0]).item(),
            dropout_p=0.0, causal=True,
        )
    else:
        output = zigzag_llama3_flash_attn_varlen_kvpacked_func(
            q_copy, kv_copy, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=q.shape[0] // (len(cu_seqlens_q) - 1),
            max_seqlen_k=(cu_seqlens_k[1] - cu_seqlens_k[0]).item(),
            dropout_p=0.0, causal=True, use_triton_kernel=True,
        )
    torch.cuda.nvtx.range_pop()

    grad_output = torch.randn_like(output)
    torch.cuda.nvtx.range_push(f"{name}_backward")
    output.backward(grad_output)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description='Profile Zigzag Ring vs Llama3')
    parser.add_argument('--mode', type=str, default='pytorch', choices=['pytorch', 'nsight'],
                       help='Profiling mode: pytorch (for flamegraphs) or nsight (for Nsight Systems)')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--nheads', type=int, default=32, help='Number of heads')
    parser.add_argument('--d', type=int, default=128, help='Head dimension')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    args = parser.parse_args()

    # Setup
    rank, world_size = setup_distributed()

    print(f"[Rank {rank}] Starting profiled benchmark...")
    print(f"  Mode: {args.mode}")
    print(f"  World size: {world_size}")
    print(f"  Sequence length: {args.seqlen}")
    print(f"  Batch size: {args.batch}")

    # Create output directory
    if rank == 0:
        os.makedirs('./profiler_logs', exist_ok=True)

    # Generate data
    q, kv, cu_seqlens_q, cu_seqlens_k = generate_test_data(
        rank, world_size, args.seqlen, args.batch, args.nheads, args.d
    )

    if args.mode == 'pytorch':
        # Run PyTorch profiler (generates flamegraphs and Chrome traces)
        print(f"\n[Rank {rank}] Running PyTorch profiler...")

        run_profiled_benchmark(
            benchmark_zigzag_ring, "zigzag_ring",
            q, kv, cu_seqlens_q, cu_seqlens_k,
            rank, world_size, iterations=args.iterations
        )

        run_profiled_benchmark(
            benchmark_zigzag_llama3, "zigzag_llama3",
            q, kv, cu_seqlens_q, cu_seqlens_k,
            rank, world_size, iterations=args.iterations
        )

        if rank == 0:
            print("\n" + "="*80)
            print("PROFILING COMPLETE!")
            print("="*80)
            print("\nVisualization Options:")
            print("1. TensorBoard: tensorboard --logdir=./profiler_logs")
            print("2. Chrome Trace: Open chrome://tracing and load *_trace_rank0.json")
            print("3. Flamegraph: flamegraph.pl *_stacks_rank0.txt > flamegraph.svg")
            print("   (Install flamegraph: git clone https://github.com/brendangregg/FlameGraph)")

    elif args.mode == 'nsight':
        # Run with NVTX markers for Nsight Systems
        print(f"\n[Rank {rank}] Running with NVTX markers for Nsight Systems...")
        print("Start Nsight Systems profiler BEFORE running this!")
        print("Command: nsys profile -o zigzag_profile python benchmark_with_profiling.py --mode nsight")

        run_nsight_benchmark(
            benchmark_zigzag_ring, "zigzag_ring",
            q, kv, cu_seqlens_q, cu_seqlens_k,
            rank, world_size
        )

        run_nsight_benchmark(
            benchmark_zigzag_llama3, "zigzag_llama3",
            q, kv, cu_seqlens_q, cu_seqlens_k,
            rank, world_size
        )

        if rank == 0:
            print("\n" + "="*80)
            print("NVTX MARKERS COMPLETE!")
            print("="*80)
            print("Open the .nsys-rep file in Nsight Systems GUI for timeline visualization")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
