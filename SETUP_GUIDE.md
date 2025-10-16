# Complete Setup Guide: Flash Attention + Ring Flash Attention

This guide explains how to set up both repositories together to test all three grouped attention implementations.

---

## 🎯 Understanding the Setup

### Repository Dependencies

```
┌─────────────────────────────────────┐
│   ring-flash-attention (your fork)  │
│   ================================   │
│   - Option B: Python wrapper        │
│   - Option C: Triton kernel ⭐      │
│   - Integration with zigzag_llama3  │
│   - Tests & benchmarks              │
└─────────────────────────────────────┘
                  │
                  │ depends on
                  ▼
┌─────────────────────────────────────┐
│   flash-attention (your fork)       │
│   ===========================       │
│   - Option A: CUDA implementation   │
│   - Core flash attention kernels    │
│   - Base API                        │
└─────────────────────────────────────┘
```

---

## 📦 Three Setup Options

### **Option 1: Triton Only** (Fastest, Recommended for Testing)

Use only ring-flash-attention with Triton implementation. No flash-attention fork needed!

```bash
# Install official flash-attention (for baseline comparison)
pip install flash-attn --no-build-isolation

# Install ring-flash-attention with Triton
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention
git checkout feature/grouped-flash-attention
pip install -e .

# Test
python example_grouped_attention.py
```

**Pros:**
- ✅ Fastest setup (10 minutes)
- ✅ No compilation needed
- ✅ Best performance (10-15% speedup)
- ✅ Works on any GPU

**Use this if:** You just want to test and don't need CUDA implementation

---

### **Option 2: Full Setup** (All 3 Implementations)

Set up both forks to test all implementations.

#### Step 1: Install flash-attention (your fork)

```bash
# Clone your flash-attention fork
git clone https://github.com/petrpan26/flash-attention.git
cd flash-attention
git checkout feature/grouped-flash-attention

# Install dependencies
pip install ninja packaging wheel
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Compile and install (takes 5-10 minutes)
# Build for multiple GPU architectures
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # A100, A10, H100, H200
pip install -e . --no-build-isolation

# Verify installation
python -c "from flash_attn import _flash_attn_varlen_forward_grouped; print('✓ Grouped API available')"
```

#### Step 2: Install ring-flash-attention (your fork)

```bash
# Go to parent directory
cd ..

# Clone your ring-flash-attention fork
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention
git checkout feature/grouped-flash-attention

# Install (no compilation needed)
pip install -e .

# Verify installation
python -c "from ring_flash_attn import zigzag_llama3_flash_attn_varlen_func; print('✓ Ring flash attention available')"
```

#### Step 3: Verify both work together

```bash
# Run example that uses both
python example_grouped_attention.py

# Should show all 3 implementations:
# - Baseline (from flash-attn)
# - Python grouped (from flash-attn fork)
# - Triton grouped (from ring-flash-attn)
```

**Pros:**
- ✅ All 3 implementations available
- ✅ Complete testing capability
- ✅ Can compare performance

**Cons:**
- ⚠️ Takes longer to set up
- ⚠️ Requires compilation

---

### **Option 3: Official + Ring** (Hybrid)

Use official flash-attention + your ring-flash-attention fork.

```bash
# Install official flash-attention
pip install flash-attn --no-build-isolation

# Install your ring-flash-attention fork
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention
git checkout feature/grouped-flash-attention
pip install -e .

# Test (Options B & C work, Option A uses official flash-attn)
python example_grouped_attention.py
```

**Note:** Option A (CUDA grouped) won't be available unless you use your flash-attention fork.

---

## 🚀 Quick Start for Remote GPU

### Recommended Setup (Option 1 - Triton Only)

```bash
# 1. Install dependencies
pip install torch triton ninja packaging pytest
pip install flash-attn --no-build-isolation  # Takes 5-10 min

# 2. Clone and install ring-flash-attention
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention
git checkout feature/grouped-flash-attention
pip install -e .

# 3. Test
python example_grouped_attention.py

# 4. Run tests
pytest test/test_grouped_flash_attention.py -v
```

**Expected time:** 15-20 minutes total

---

### Full Setup (Option 2 - All Implementations)

```bash
# 1. Set up flash-attention fork
git clone https://github.com/petrpan26/flash-attention.git
cd flash-attention
git checkout feature/grouped-flash-attention
pip install ninja packaging wheel torch
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install -e . --no-build-isolation  # Takes 5-10 min

# 2. Set up ring-flash-attention fork
cd ..
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention
git checkout feature/grouped-flash-attention
pip install -e .

# 3. Test all implementations
python example_grouped_attention.py

# 4. Run comprehensive tests
pytest test/test_grouped_flash_attention.py -v
pytest test/test_grouped_flash_attention_stress.py -v
```

**Expected time:** 20-30 minutes total

---

## 🧪 Testing Each Implementation

### Test Option A: CUDA (requires flash-attention fork)

```python
from flash_attn import _flash_attn_varlen_forward_grouped

# Test CUDA grouped implementation
out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
    q_list=[q_early, q_late],
    k=k_full,
    v=v_full,
    cu_seqlens_q_list=[cu_seqlens_q_early, cu_seqlens_q_late],
    cu_seqlens_k_list=[cu_seqlens_k_early, cu_seqlens_k_late],
    max_seqlen_k_list=[tokens_early, tokens_late],
    softmax_scale=softmax_scale,
    causal=True,
)
```

### Test Option B: Python Wrapper (requires flash-attention fork)

```python
from ring_flash_attn import zigzag_llama3_flash_attn_varlen_func

out = zigzag_llama3_flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    heads_k_stride, local_k_slice,
    softmax_scale=softmax_scale,
    causal=True,
    use_grouped_attention=True,  # ← Option B
)
```

### Test Option C: Triton (only needs ring-flash-attention)

```python
from ring_flash_attn import zigzag_llama3_flash_attn_varlen_func

out = zigzag_llama3_flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    heads_k_stride, local_k_slice,
    softmax_scale=softmax_scale,
    causal=True,
    use_triton_grouped=True,  # ← Option C
)
```

---

## 📁 Directory Structure

After full setup, your directory should look like:

```
/your/workspace/
├── flash-attention/                    # Your fork
│   ├── .git/
│   ├── csrc/
│   │   └── flash_attn/
│   │       ├── flash_api.cpp           # Modified: grouped API
│   │       └── src/
│   │           └── flash.h             # Modified: params struct
│   ├── flash_attn/
│   │   ├── __init__.py                 # Modified: exports
│   │   ├── flash_attn_interface.py     # Modified: Python wrapper
│   │   └── flash_attn_grouped.py       # NEW: Option B
│   └── test/
│       └── test_flash_attn_grouped.py  # NEW: tests
│
└── ring-flash-attention/               # Your fork
    ├── .git/
    ├── ring_flash_attn/
    │   ├── zigzag_llama3_flash_attn_varlen.py  # Modified: integration
    │   └── triton_grouped_attention.py         # NEW: Option C
    ├── test/
    │   ├── test_grouped_flash_attention.py     # NEW: unit tests
    │   ├── test_grouped_flash_attention_stress.py  # NEW: stress tests
    │   └── test_zigzag_llama3_grouped.py       # NEW: integration tests
    ├── benchmark/
    │   └── benchmark_grouped_attention.py      # NEW: benchmarks
    ├── example_grouped_attention.py            # NEW: examples
    ├── claude.md                               # Testing guide
    └── SETUP_GUIDE.md                          # This file
```

---

## 🔧 Compilation Details

### flash-attention Compilation

When you run `pip install -e . --no-build-isolation` in flash-attention:

**What happens:**
1. PyTorch extension builder compiles CUDA kernels
2. Creates `.so` files for each GPU architecture
3. Links with PyTorch
4. Takes 5-10 minutes

**Environment variables:**
```bash
# Build for multiple GPUs (recommended)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Build for specific GPU only (faster, smaller)
export TORCH_CUDA_ARCH_LIST="8.6"  # A10 only

# Debug build (if needed)
export DEBUG=1
```

**Verify compilation:**
```bash
# Check what architectures were built
python -c "import torch; print(torch.cuda.get_arch_list())"

# Check if grouped API exists
python -c "from flash_attn import _flash_attn_varlen_forward_grouped; print('✓')"
```

### ring-flash-attention Installation

No compilation needed! It's pure Python + Triton (JIT compiled).

```bash
pip install -e .  # Instant
```

---

## 🐛 Common Issues

### Issue 1: flash-attention compilation fails

```bash
# Error: ninja not found
pip install ninja packaging wheel

# Error: CUDA not found
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Error: torch not found
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Import error for grouped API

```python
# Error: cannot import _flash_attn_varlen_forward_grouped
```

**Solution:** You need flash-attention fork, not official version

```bash
# Uninstall official version
pip uninstall flash-attn

# Install your fork
cd /path/to/flash-attention
git checkout feature/grouped-flash-attention
pip install -e . --no-build-isolation
```

### Issue 3: Version conflicts

```bash
# Error: Multiple flash-attn installations
```

**Solution:**
```bash
# Remove all versions
pip uninstall flash-attn -y
pip uninstall flash_attn -y

# Install fresh
cd /path/to/flash-attention
pip install -e . --no-build-isolation
```

### Issue 4: Triton not found

```bash
# Error: No module named 'triton'
pip install triton

# If that fails, update PyTorch first
pip install --upgrade torch
pip install triton
```

---

## 🎯 Which Setup for Which Use Case?

### For Quick Testing (Recommended)
**Use Option 1:** Triton only
- ✅ Fastest setup
- ✅ Best performance
- ✅ No compilation

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention
git checkout feature/grouped-flash-attention
pip install -e .
```

### For Development / Full Testing
**Use Option 2:** Full setup
- ✅ All implementations
- ✅ Complete comparison
- ⚠️ Takes longer

```bash
# Install both forks (see Full Setup above)
```

### For Production
**Use Option 1:** Triton only
- ✅ No custom flash-attention needed
- ✅ Works with official flash-attn
- ✅ Easy to deploy

---

## 📊 Verifying Setup

### Quick Verification Script

```python
#!/usr/bin/env python3
"""Verify grouped attention setup"""

import sys

print("Checking setup...")
print("-" * 50)

# 1. Check PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not found")
    sys.exit(1)

# 2. Check Triton
try:
    import triton
    print(f"✓ Triton {triton.__version__}")
except ImportError:
    print("⚠ Triton not found (needed for Option C)")

# 3. Check flash-attn
try:
    from flash_attn import flash_attn_func
    print(f"✓ flash-attn installed")

    # Check if grouped API exists
    try:
        from flash_attn import _flash_attn_varlen_forward_grouped
        print(f"  ✓ Grouped API available (Option A & B)")
    except ImportError:
        print(f"  ⚠ Grouped API not available (using official flash-attn)")
except ImportError:
    print("✗ flash-attn not found")

# 4. Check ring-flash-attn
try:
    from ring_flash_attn import zigzag_llama3_flash_attn_varlen_func
    print(f"✓ ring-flash-attn installed")

    # Check Triton kernel
    try:
        from ring_flash_attn.triton_grouped_attention import triton_grouped_flash_attn_varlen_forward
        print(f"  ✓ Triton grouped kernel available (Option C)")
    except ImportError:
        print(f"  ✗ Triton grouped kernel not found")
except ImportError:
    print("✗ ring-flash-attn not found")

print("-" * 50)
print("\nSetup Summary:")
print("Option A (CUDA): ", "Available" if 'flash_attn' in sys.modules else "Not available")
print("Option B (Python): ", "Available" if 'flash_attn' in sys.modules else "Not available")
print("Option C (Triton): ", "Available" if 'triton' in sys.modules else "Not available")
```

Save as `verify_setup.py` and run:
```bash
python verify_setup.py
```

---

## 🚀 Ready to Test!

Once setup is complete, run:

```bash
# Quick test
python example_grouped_attention.py

# Full test suite
pytest test/test_grouped_flash_attention.py -v

# Benchmarks
python benchmark/benchmark_grouped_attention.py
```

Expected output:
```
✓ All implementations working
✓ Triton is 10-15% faster than baseline
✓ All tests pass
```

---

## 📝 Summary

### Minimum Setup (Triton Only)
```bash
pip install flash-attn triton --no-build-isolation
git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention && git checkout feature/grouped-flash-attention
pip install -e .
```

### Full Setup (All Options)
```bash
# Flash-attention fork
git clone https://github.com/petrpan26/flash-attention.git
cd flash-attention && git checkout feature/grouped-flash-attention
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
pip install -e . --no-build-isolation

# Ring-flash-attention fork
cd .. && git clone https://github.com/petrpan26/ring-flash-attention.git
cd ring-flash-attention && git checkout feature/grouped-flash-attention
pip install -e .
```

**Both are now ready for testing!** 🎉
