from flash_attn import flash_attn_varlen_kvpacked_func
import os
import sys
import torch
import torch.distributed as dist
from ring_flash_attn import (
    ring_flash_attn_varlen_kvpacked_func,
    zigzag_ring_flash_attn_varlen_kvpacked_func,
    llama3_flash_attn_varlen_kvpacked_func,
    zigzag_llama3_flash_attn_varlen_kvpacked_func,
    llama3_flash_attn_prepare_cu_seqlens,
)


def benchmark(
    f,
    use_double_cu_seqlens,
    use_llama3=False,
    num_iter=100,
    forward_only=True,
    log=True,
    profile=False,
):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    deterministic = False
    # config of llama3 8B
    seqlen = 1024 * 8
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    causal = True

    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0

    q = torch.randn(
        seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.randn(
        seqlen,
        2,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_list = [
        torch.tensor([0, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 256, 7648, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 4096, 8192], device=device, dtype=torch.int32),
        torch.tensor(
            [0, 3104, 6304, 7904, 8064, 8192], device=device, dtype=torch.int32
        ),
    ]

    if use_llama3:
        cu_seqlens_q_list = []
        cu_seqlens_k_list = []
        max_seqlen_q_list = []
        max_seqlen_k_list = []
        local_k_slice_list = []
        for cu_seqlens in cu_seqlens_list:
            (
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                local_k_slice,
            ) = llama3_flash_attn_prepare_cu_seqlens(
                cu_seqlens * world_size,
                causal=causal,
                rank=rank,
                world_size=world_size,
            )
            cu_seqlens_q_list.append(cu_seqlens_q)
            cu_seqlens_k_list.append(cu_seqlens_k)
            max_seqlen_q_list.append(max_seqlen_q)
            max_seqlen_k_list.append(max_seqlen_k)
            local_k_slice_list.append(local_k_slice)
    else:
        max_seqlen_list = [
            (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            for cu_seqlens in cu_seqlens_list
        ]

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    def wrapper(i: int):
        if use_llama3:
            return f(
                q,
                kv,
                cu_seqlens_q_list[i % len(cu_seqlens_list)],
                cu_seqlens_k_list[i % len(cu_seqlens_list)],
                max_seqlen_q_list[i % len(cu_seqlens_list)],
                max_seqlen_k_list[i % len(cu_seqlens_list)],
                heads_k_stride=4,
                local_k_slice=local_k_slice_list[i % len(cu_seqlens_list)],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
        elif use_double_cu_seqlens:
            return f(
                q,
                kv,
                cu_seqlens_list[i % len(cu_seqlens_list)],
                cu_seqlens_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(cu_seqlens_list)],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
        else:
            return f(
                q,
                kv,
                cu_seqlens_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(cu_seqlens_list)],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )

    if forward_only:
        with torch.no_grad():
            for i in range(num_iter):
                _ = wrapper(i)
    else:
        for i in range(num_iter):
            q.grad = None
            kv.grad = None
            out = wrapper(i)
            out.backward(dout)
            if profile:
                profiler.step()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()
    if rank == 0 and log:
        print(f"{num_iter / time} iter/s, {time} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = False
    profile = False
    compile_func = False

    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg == "compile":
            compile_func = True
            torch._dynamo.config.capture_scalar_outputs = True
        elif arg == "forward_only":
            forward_only = True
        elif arg == "profile":
            profile = True

    num_iter = 500 if forward_only else 100

    for f, use_double_cu_seqlens in [
        (flash_attn_varlen_kvpacked_func, True),
        (ring_flash_attn_varlen_kvpacked_func, False),
        (zigzag_ring_flash_attn_varlen_kvpacked_func, False),
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        f = torch.compile(f) if compile_func else f
        benchmark(
            f,
            use_double_cu_seqlens,
            forward_only=forward_only,
            num_iter=num_iter,
            log=False,
        )
        benchmark(
            f,
            use_double_cu_seqlens,
            forward_only=forward_only,
            num_iter=num_iter,
            log=True,
            profile=profile,
        )

    for f, use_double_cu_seqlens in [
        (llama3_flash_attn_varlen_kvpacked_func, True),
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        f = torch.compile(f) if compile_func else f
        benchmark(
            f,
            use_double_cu_seqlens,
            use_llama3=True,
            forward_only=forward_only,
            num_iter=num_iter,
            log=False,
        )
        benchmark(
            f,
            use_double_cu_seqlens,
            use_llama3=True,
            forward_only=forward_only,
            num_iter=num_iter,
            log=True,
            profile=profile,
        )

    # Benchmark zigzag_llama3_flash_attn with all 4 kernel mode combinations
    # This requires special data preparation (zigzag extraction)
    def extract_local_zigzag(value, cu_seqlens, rank, world_size):
        """Extract local zigzag-distributed portion for this rank."""
        local_values = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            local_value = value[start:end].chunk(2 * world_size, dim=0)
            local_values.extend([
                local_value[rank].detach().clone(),
                local_value[2 * world_size - 1 - rank].detach().clone(),
            ])
        return torch.cat(local_values, dim=0).contiguous()

    # Prepare data for zigzag_llama3
    # To match other benchmarks: each rank processes 8192 tokens
    # So total tokens = 8192 * world_size
    seqlen_per_rank = 1024 * 8
    world_size = dist.get_world_size()
    total_seqlen = seqlen_per_rank * world_size
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    # Create full sequences and broadcast
    q_full = torch.randn(total_seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    kv_full = torch.randn(total_seqlen, 2, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    dout_full = torch.randn(total_seqlen, num_heads, head_dim, device=device, dtype=dtype)

    dist.broadcast(q_full, src=0)
    dist.broadcast(kv_full, src=0)
    dist.broadcast(dout_full, src=0)

    cu_seqlens_single = torch.tensor([0, total_seqlen], device=device, dtype=torch.int32)

    # Extract local zigzag portions
    local_q = extract_local_zigzag(q_full, cu_seqlens_single, rank, dist.get_world_size())
    local_kv = extract_local_zigzag(kv_full, cu_seqlens_single, rank, dist.get_world_size())
    local_dout = extract_local_zigzag(dout_full, cu_seqlens_single, rank, dist.get_world_size())
    local_q.requires_grad = True
    local_kv.requires_grad = True

    # Prepare local cu_seqlens for Q
    # Each rank gets seqlen_per_rank tokens after zigzag extraction
    local_cu_seqlens_q = torch.tensor([0, seqlen_per_rank], dtype=torch.int32, device=device)

    # Prepare parameters
    _, _, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_single, True, rank, dist.get_world_size()
    )

    for use_fused_fwd, use_fused_bwd, mode_name in [
        (False, False, "Two-Kernels Forward, Two-Kernels Backward (Triton Optimized)"),
        (False, True, "Two-Kernels Forward, Fused Backward (Triton Optimized)"),
        (True, False, "Fused Forward, Two-Kernels Backward (Python Fallback)"),
        (True, True, "Fused Forward, Fused Backward (Python Fallback)"),
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# zigzag_llama3_flash_attn_varlen_kvpacked_func ({mode_name})")

        # Warmup
        with torch.no_grad() if forward_only else torch.enable_grad():
            for _ in range(10):
                local_q.grad = None if not forward_only else None
                local_kv.grad = None if not forward_only else None
                out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
                    local_q, local_kv,
                    local_cu_seqlens_q,
                    cu_seqlens_single,
                    max_seqlen_q, max_seqlen_k,
                    heads_k_stride=num_kv_heads,
                    local_k_slice=local_k_slice,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    use_fused_kernel_forward=use_fused_fwd,
                    use_fused_kernel_backward=use_fused_bwd,
                    n_chunks=2,
                )
                if not forward_only:
                    out.backward(local_dout)

        # Actual benchmark
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()

        if forward_only:
            with torch.no_grad():
                for _ in range(num_iter):
                    out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
                        local_q, local_kv,
                        local_cu_seqlens_q,
                        cu_seqlens_single,
                        max_seqlen_q, max_seqlen_k,
                        heads_k_stride=num_kv_heads,
                        local_k_slice=local_k_slice,
                        causal=True,
                        window_size=(-1, -1),
                        alibi_slopes=None,
                        deterministic=False,
                        return_attn_probs=False,
                        use_fused_kernel_forward=use_fused_fwd,
                        use_fused_kernel_backward=use_fused_bwd,
                        n_chunks=2,
                    )
        else:
            for _ in range(num_iter):
                local_q.grad = None
                local_kv.grad = None
                out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
                    local_q, local_kv,
                    local_cu_seqlens_q,
                    cu_seqlens_single,
                    max_seqlen_q, max_seqlen_k,
                    heads_k_stride=num_kv_heads,
                    local_k_slice=local_k_slice,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    use_fused_kernel_forward=use_fused_fwd,
                    use_fused_kernel_backward=use_fused_bwd,
                    n_chunks=2,
                )
                out.backward(local_dout)

        end.record()
        torch.cuda.synchronize(device=device)
        time = begin.elapsed_time(end) / 1000.0

        if rank == 0:
            print(f"{num_iter / time} iter/s, {time} sec")

    dist.destroy_process_group()
