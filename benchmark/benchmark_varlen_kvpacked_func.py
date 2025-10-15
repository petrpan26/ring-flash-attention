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
    use_zigzag_llama3=False,
    use_fused_fwd=False,
    use_fused_bwd=False,
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

    if use_zigzag_llama3:
        # For zigzag_llama3: broadcast data and extract local zigzag portions
        def extract_local_zigzag(value, cu_seqlens, rank, world_size):
            """Extract local zigzag-distributed portion for this rank."""
            local_values = []
            for i in range(len(cu_seqlens) - 1):
                start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                seq_len = end - start
                assert seq_len % (2 * world_size) == 0, \
                    f"Sequence {i} length {seq_len} not divisible by 2*world_size={2*world_size}"
                seq_data = value[start:end]
                local_chunks = seq_data.chunk(2 * world_size, dim=0)
                local_values.extend([
                    local_chunks[rank].detach().clone(),
                    local_chunks[2 * world_size - 1 - rank].detach().clone(),
                ])
            return torch.cat(local_values, dim=0).contiguous()

        # Broadcast full sequences
        dist.broadcast(q, src=0)
        dist.broadcast(kv, src=0)
        dist.broadcast(dout, src=0)

        # Prepare parameters for each cu_seqlens configuration
        zigzag_q_list = []
        zigzag_kv_list = []
        zigzag_dout_list = []
        cu_seqlens_q_list = []
        cu_seqlens_k_list = []
        max_seqlen_q_list = []
        max_seqlen_k_list = []
        local_k_slice_list = []

        for cu_seqlens in cu_seqlens_list:
            # Extract local zigzag portions
            local_q = extract_local_zigzag(q, cu_seqlens, rank, world_size)
            local_kv = extract_local_zigzag(kv, cu_seqlens, rank, world_size)
            local_dout = extract_local_zigzag(dout, cu_seqlens, rank, world_size)
            local_q.requires_grad = True
            local_kv.requires_grad = True

            zigzag_q_list.append(local_q)
            zigzag_kv_list.append(local_kv)
            zigzag_dout_list.append(local_dout)

            # Calculate local cu_seqlens_q
            local_seqlens = []
            offset = 0
            for i in range(len(cu_seqlens) - 1):
                seq_len = (cu_seqlens[i+1] - cu_seqlens[i]).item()
                local_seq_len = seq_len // world_size
                local_seqlens.append(offset)
                offset += local_seq_len
            local_seqlens.append(offset)
            local_cu_seqlens_q = torch.tensor(local_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_q_list.append(local_cu_seqlens_q)

            # Global cu_seqlens_k and prepare llama3 parameters
            cu_seqlens_k = cu_seqlens * world_size
            _, _, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
                cu_seqlens_k, causal, rank, world_size
            )
            cu_seqlens_k_list.append(cu_seqlens_k)
            max_seqlen_q_list.append(max_seqlen_q)
            max_seqlen_k_list.append(max_seqlen_k)
            local_k_slice_list.append(local_k_slice)

    elif use_llama3:
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
        if use_zigzag_llama3:
            idx = i % len(cu_seqlens_list)
            return f(
                zigzag_q_list[idx],
                zigzag_kv_list[idx],
                cu_seqlens_q_list[idx],
                cu_seqlens_k_list[idx],
                max_seqlen_q_list[idx],
                max_seqlen_k_list[idx],
                heads_k_stride=num_kv_heads,
                local_k_slice=local_k_slice_list[idx],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
                use_fused_kernel_forward=use_fused_fwd,
                use_fused_kernel_backward=use_fused_bwd,
                n_chunks=2,
            )
        elif use_llama3:
            return f(
                q,
                kv,
                cu_seqlens_q_list[i % len(cu_seqlens_list)],
                cu_seqlens_k_list[i % len(cu_seqlens_list)],
                max_seqlen_q_list[i % len(cu_seqlens_list)],
                max_seqlen_k_list[i % len(cu_seqlens_list)],
                heads_k_stride=1,
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
            if use_zigzag_llama3:
                idx = i % len(cu_seqlens_list)
                zigzag_q_list[idx].grad = None
                zigzag_kv_list[idx].grad = None
                out = wrapper(i)
                out.backward(zigzag_dout_list[idx])
            else:
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
    # Uses the shared benchmark() function with use_zigzag_llama3=True
    for fused_fwd, fused_bwd, mode_name in [
        (False, False, "Two-Kernels Forward, Two-Kernels Backward (Triton Optimized)"),
        (False, True, "Two-Kernels Forward, Fused Backward (Triton Optimized)"),
        (True, False, "Fused Forward, Two-Kernels Backward (Python Fallback)"),
        (True, True, "Fused Forward, Fused Backward (Python Fallback)"),
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# zigzag_llama3_flash_attn_varlen_kvpacked_func ({mode_name})")

        benchmark(
            zigzag_llama3_flash_attn_varlen_kvpacked_func,
            use_double_cu_seqlens=True,
            use_zigzag_llama3=True,
            use_fused_fwd=fused_fwd,
            use_fused_bwd=fused_bwd,
            forward_only=forward_only,
            num_iter=num_iter,
            log=False,
        )
        benchmark(
            zigzag_llama3_flash_attn_varlen_kvpacked_func,
            use_double_cu_seqlens=True,
            use_zigzag_llama3=True,
            use_fused_fwd=fused_fwd,
            use_fused_bwd=fused_bwd,
            forward_only=forward_only,
            num_iter=num_iter,
            log=True,
            profile=profile,
        )

    dist.destroy_process_group()
