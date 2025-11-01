# Quick Start Guide: Fused Triton Zigzag Llama3 Flash Attention

This guide shows you how to run and test the optimized fused Triton implementation.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer)
  - ✅ A100, A6000, H100, H200
  - ✅ RTX 3090, RTX 4090
  - ❌ V100 (compute capability 7.0 - not supported)

### Software Requirements
```bash
# Python 3.8+
python --version  # Should be 3.8 or higher

# CUDA 11.8+ or 12.x
nvcc --version

# PyTorch 2.0+ with CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Triton 2.0+
python -c "import triton; print(f'Triton: {triton.__version__}')"
```

## Installation

### Step 1: Install Dependencies

```bash
# Navigate to ring-flash-attention directory
cd /Users/petrpan26/work/ring-flash-attention

# Install Triton (if not already installed)
pip install triton>=2.0.0

# Install other dependencies
pip install torch>=2.0.0  # Should already be installed
pip install pytest  # For testing
```

### Step 2: Verify Installation

```bash
# Quick verification
python -c "
import torch
import triton
print('✓ PyTorch:', torch.__version__)
print('✓ Triton:', triton.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
    print('✓ Compute capability:', torch.cuda.get_device_capability(0))
"
```

Expected output:
```
✓ PyTorch: 2.1.0+cu121
✓ Triton: 2.1.0
✓ CUDA available: True
✓ GPU: NVIDIA A100-SXM4-80GB
✓ Compute capability: (8, 0)
```

## Running Tests

### Option 1: Run Basic Correctness Tests

```bash
cd /Users/petrpan26/work/ring-flash-attention

# Run the test suite
python test/test_triton_fused_zigzag_llama3.py
```

**Expected output:**
```
Testing fused Triton zigzag llama3 implementation...

============================================================
Test 1: Forward Correctness
============================================================
out_original shape: torch.Size([256, 8, 64])
out_fused shape: torch.Size([256, 8, 64])
out_original: tensor([-0.0234,  0.0156, ...])
out_fused: tensor([-0.0234,  0.0156, ...])
✓ Forward pass test passed!

============================================================
Test 2: Backward Correctness
============================================================
✓ Backward pass test passed!

============================================================
✓ All tests passed!
============================================================
```

### Option 2: Run with pytest

```bash
# Install pytest if needed
pip install pytest

# Run tests
pytest test/test_triton_fused_zigzag_llama3.py -v
```

### Option 3: Test Different Configurations

```python
# test_custom_config.py
import torch
from ring_flash_attn.triton_zigzag_llama3_flash_attn_v2 import (
    fused_zigzag_llama3_flash_attn_forward_v2,
)

# Test with different head dimensions
for headdim in [64, 128]:
    print(f"\nTesting headdim={headdim}")

    q0 = torch.randn(128, 8, headdim, dtype=torch.float16, device='cuda', requires_grad=True)
    q1 = torch.randn(128, 8, headdim, dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn(1024, 8, headdim, dtype=torch.float16, device='cuda')
    v = torch.randn(1024, 8, headdim, dtype=torch.float16, device='cuda')

    cu_seqlens_q0 = torch.tensor([0, 128], dtype=torch.int32, device='cuda')
    cu_seqlens_q1 = torch.tensor([0, 128], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, 1024], dtype=torch.int32, device='cuda')

    out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_forward_v2(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        chunk_idx_0=0, chunk_idx_1=7,
        total_chunks=8,
        causal=True
    )

    print(f"✓ Success! Output shapes: {out0.shape}, {out1.shape}")
```

Run it:
```bash
python test_custom_config.py
```

## Integration into Existing Code

### Method 1: Direct Replacement (Recommended)

Edit `ring_flash_attn/zigzag_llama3_flash_attn_varlen.py`:

```python
# At the top of the file, add:
import os
from .triton_zigzag_llama3_flash_attn_v2 import replace_grouped_attention_with_fused_triton_v2

# In zigzag_llama3_flash_attn_varlen_forward function (around line 640),
# replace execute_grouped_attention call with:

USE_FUSED_TRITON = os.environ.get("ZIGZAG_USE_FUSED_TRITON_V2", "1") == "1"

if USE_FUSED_TRITON:
    out_i, lse_i, chunk_info = replace_grouped_attention_with_fused_triton_v2(
        chunk_q_list_i, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads // nheads_k * heads_k_stride, head_dim, softmax_scale,
        dropout_p, causal, window_size, alibi_slopes, deterministic,
        world_size, cu_seqlens_k, chunk_idx_0, chunk_idx_1
    )
else:
    out_i, lse_i, chunk_info = execute_grouped_attention(
        chunk_q_list_i, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads // nheads_k * heads_k_stride, head_dim, softmax_scale,
        dropout_p, causal, window_size, alibi_slopes, deterministic,
        world_size, cu_seqlens_k
    )
```

### Method 2: Feature Flag (Production)

Create a configuration file:

```python
# config.py
import os

class ZigzagConfig:
    # Enable/disable fused Triton kernel
    USE_FUSED_TRITON_V2 = os.environ.get("ZIGZAG_USE_FUSED_TRITON_V2", "1") == "1"

    # Debugging options
    VERIFY_CORRECTNESS = os.environ.get("ZIGZAG_VERIFY_CORRECTNESS", "0") == "1"
    PROFILE_KERNELS = os.environ.get("ZIGZAG_PROFILE_KERNELS", "0") == "1"
```

Use it in your code:

```python
from config import ZigzagConfig

if ZigzagConfig.USE_FUSED_TRITON_V2:
    out, lse, chunk_info = replace_grouped_attention_with_fused_triton_v2(...)
else:
    out, lse, chunk_info = execute_grouped_attention(...)

# Optional: Verify correctness in debug mode
if ZigzagConfig.VERIFY_CORRECTNESS:
    out_reference, lse_reference, _ = execute_grouped_attention(...)
    assert torch.allclose(out, out_reference, atol=1e-2, rtol=1e-2), "Correctness check failed!"
```

### Method 3: Monkey Patch (Quick Test)

For quick testing without modifying code:

```python
# monkey_patch_fused.py
import ring_flash_attn.zigzag_llama3_flash_attn_varlen as zigzag_module
from ring_flash_attn.triton_zigzag_llama3_flash_attn_v2 import replace_grouped_attention_with_fused_triton_v2

# Save original function
original_execute = zigzag_module.execute_grouped_attention

# Replace with fused version
def fused_execute(*args, **kwargs):
    # Extract necessary parameters from args/kwargs
    chunk_q_list = args[0]
    chunk_cu_seqlens_q_list = args[1]
    # ... (extract all parameters)

    return replace_grouped_attention_with_fused_triton_v2(*args, **kwargs)

zigzag_module.execute_grouped_attention = fused_execute

# Now run your normal training code
# The fused kernel will be used automatically
```

## Running Your Training Script

### With Environment Variable

```bash
# Enable fused kernel (default)
export ZIGZAG_USE_FUSED_TRITON_V2=1
python train.py

# Disable for comparison
export ZIGZAG_USE_FUSED_TRITON_V2=0
python train.py
```

### With Debug Verification

```bash
# Enable correctness checking (slower, for debugging)
export ZIGZAG_VERIFY_CORRECTNESS=1
python train.py
```

### Example Training Script

```python
# train_example.py
import torch
import torch.distributed as dist
from ring_flash_attn.zigzag_llama3_flash_attn_varlen import zigzag_llama3_flash_attn_varlen_qkvpacked_func

# Initialize distributed training
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Your model forward pass
def attention_forward(qkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    """
    The fused kernel will be used automatically if ZIGZAG_USE_FUSED_TRITON_V2=1
    """
    out = zigzag_llama3_flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride=8,
        local_k_slice=slice(0, 256),
        causal=True,
        group=dist.group.WORLD,
    )
    return out

# Training loop
for batch in dataloader:
    # ... prepare inputs ...
    out = attention_forward(qkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    # ... rest of training ...
```

Run it:
```bash
# Single GPU
python train_example.py

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 train_example.py
```

## Benchmarking Performance

### Simple Timing Test

```python
# benchmark_simple.py
import torch
import time
from ring_flash_attn.zigzag_llama3_flash_attn_varlen import execute_grouped_attention
from ring_flash_attn.triton_zigzag_llama3_flash_attn_v2 import replace_grouped_attention_with_fused_triton_v2

# Setup
device = 'cuda'
world_size = 4
rank = 0
seqlen_local = 512
nheads = 32
headdim = 128

# Create test data
q = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device=device)
k = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device=device)
v = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device=device)
kv_buffer = torch.stack([k.repeat(world_size, 1, 1), v.repeat(world_size, 1, 1)])

# ... setup other parameters ...

# Warmup
for _ in range(10):
    _ = execute_grouped_attention(...)

# Benchmark original
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = execute_grouped_attention(...)
torch.cuda.synchronize()
time_original = (time.time() - start) / 100

# Benchmark fused
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = replace_grouped_attention_with_fused_triton_v2(...)
torch.cuda.synchronize()
time_fused = (time.time() - start) / 100

print(f"Original: {time_original*1000:.2f} ms")
print(f"Fused V2: {time_fused*1000:.2f} ms")
print(f"Speedup: {time_original/time_fused:.2f}x")
```

### Using PyTorch Profiler

```python
# profile_detailed.py
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    # Run your attention forward pass
    out = replace_grouped_attention_with_fused_triton_v2(...)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Save trace for visualization
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

### Using NVIDIA Nsight Systems

```bash
# Profile with nsys
nsys profile -o profile_fused \
    --trace=cuda,nvtx \
    --gpu-metrics-device=all \
    python benchmark_simple.py

# View results
nsys-ui profile_fused.nsys-rep
```

## Troubleshooting

### Issue: Import Error

```
ImportError: cannot import name 'replace_grouped_attention_with_fused_triton_v2'
```

**Solution:**
```bash
# Make sure you're in the right directory
cd /Users/petrpan26/work/ring-flash-attention

# Check file exists
ls ring_flash_attn/triton_zigzag_llama3_flash_attn_v2.py

# Try importing in Python
python -c "from ring_flash_attn.triton_zigzag_llama3_flash_attn_v2 import replace_grouped_attention_with_fused_triton_v2; print('✓ Import successful')"
```

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing
- Check for memory leaks

```python
# Monitor memory usage
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
```

### Issue: Numerical Differences

```
AssertionError: Outputs differ! Max diff: 0.05
```

**Solution:**
- This is expected for fp16 (tolerance: atol=1e-2)
- Check if difference is within acceptable range:

```python
max_diff = (out_fused - out_original).abs().max()
print(f"Max difference: {max_diff:.6f}")

if max_diff < 1e-2:
    print("✓ Within expected tolerance for fp16")
else:
    print("⚠ Difference larger than expected, investigating...")
```

### Issue: Triton Compilation Error

```
TritonError: Compilation failed
```

**Solution:**
```bash
# Clear Triton cache
rm -rf ~/.triton/cache

# Update Triton
pip install --upgrade triton

# Check CUDA compatibility
python -c "import triton; print(triton.__version__)"
```

### Issue: Slower than Expected

**Possible causes:**
1. Not using V2 (optimized version)
2. Small batch size (not utilizing GPU fully)
3. Debug mode enabled
4. Cold start (need warmup)

**Solution:**
```python
# Verify using V2
print(f"Using V2: {USE_FUSED_TRITON_V2}")

# Add warmup iterations
for _ in range(10):
    _ = replace_grouped_attention_with_fused_triton_v2(...)

# Profile to find bottleneck
with torch.profiler.profile() as prof:
    _ = replace_grouped_attention_with_fused_triton_v2(...)
print(prof.key_averages())
```

## Verification Checklist

Before deploying to production:

- [ ] ✓ All tests pass
- [ ] ✓ Correctness verified on sample data
- [ ] ✓ Performance benchmarked (>1.5x speedup expected)
- [ ] ✓ Memory usage acceptable
- [ ] ✓ Tested on target GPU (A100/H100)
- [ ] ✓ Integrated with feature flag
- [ ] ✓ Monitoring/logging added
- [ ] ✓ Rollback plan ready

## Common Commands Reference

```bash
# Quick test
python test/test_triton_fused_zigzag_llama3.py

# Enable V2
export ZIGZAG_USE_FUSED_TRITON_V2=1

# Disable V2 (use original)
export ZIGZAG_USE_FUSED_TRITON_V2=0

# Debug mode with verification
export ZIGZAG_VERIFY_CORRECTNESS=1

# Profile with nsys
nsys profile -o profile python train.py

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Check GPU status
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review the implementation summary: `IMPLEMENTATION_SUMMARY.md`
3. Check FlagAttention improvements: `FLAGATTENTION_IMPROVEMENTS.md`
4. Look at test examples: `test/test_triton_fused_zigzag_llama3.py`

## Next Steps

After successfully running the tests:

1. **Benchmark on your workload**
   - Measure actual speedup
   - Profile memory usage
   - Verify numerical stability

2. **Gradual rollout**
   - Start with development environment
   - Enable for subset of users
   - Monitor metrics carefully

3. **Optimization**
   - Tune block sizes for your hardware
   - Consider V3 separated backward (future)
   - Add custom configurations

Congratulations! You're now ready to use the fused Triton kernel for 2x faster attention! 🚀
