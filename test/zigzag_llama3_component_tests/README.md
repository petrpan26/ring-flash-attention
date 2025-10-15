# Zigzag Llama3 Component Tests

Unit tests for individual components of the `zigzag_llama3_flash_attn_varlen` implementation.

## Test Files

### `test_triton_kernels.py`
Tests for Triton kernel optimizations:
- **`extract_zigzag_kv_slice_kernel`** - Direct KV extraction from zigzag format (Optimization #1)
- **`scatter_grad_to_zigzag_kernel`** - Gradient scattering for backward pass (Optimization #5)

**Test Coverage:**
- Single sequence vs multiple sequences
- Early chunks vs late chunks vs full sequence
- Different sequence lengths and head dimensions
- Performance benchmarks (Triton vs Python reference)

### `test_zigzag_rearrange_functions.py`
Tests for zigzag rearrangement helper functions (Python reference implementations).

### `test_extract_local.py`
Tests for extracting local zigzag portions for each rank.

## Running Tests

### Run All Component Tests

```bash
# Run all component tests (requires 8 GPUs)
torchrun --nproc_per_node 8 test/zigzag_llama3_component_tests/test_triton_kernels.py
```

### Run Individual Tests

```bash
# Run just Triton kernel tests
torchrun --nproc_per_node 8 -m pytest test/zigzag_llama3_component_tests/test_triton_kernels.py -v

# Run specific test function
torchrun --nproc_per_node 8 -m pytest test/zigzag_llama3_component_tests/test_triton_kernels.py::test_extract_zigzag_kv_slices_single_sequence -v
```

### Run with Different World Sizes

```bash
# Test with 4 GPUs
torchrun --nproc_per_node 4 test/zigzag_llama3_component_tests/test_triton_kernels.py

# Test with 2 GPUs
torchrun --nproc_per_node 2 test/zigzag_llama3_component_tests/test_triton_kernels.py
```

## Test Output

Each test prints detailed comparison metrics:

```
[Rank 0] Test: extract_zigzag_kv_slices_single_sequence
  K max diff: 0.000000e+00
  K mean diff: 0.000000e+00
  V max diff: 0.000000e+00
  V mean diff: 0.000000e+00
  ✓ PASSED

[Rank 0] Performance Comparison:
  Triton kernel: 0.123 ms
  Python reference: 1.456 ms
  Speedup: 11.84x
  ✓ PASSED (Triton is faster)
```

## What Each Test Validates

### Correctness Tests

1. **`test_extract_zigzag_kv_slices_single_sequence`**
   - Validates Triton kernel produces exact same output as Python reference
   - Tests early chunk extraction (chunks 0 to world_size-1)

2. **`test_extract_zigzag_kv_slices_multiple_sequences`**
   - Tests with varying sequence lengths: [512, 1024, 768]
   - Validates correct handling of multiple sequences

3. **`test_extract_zigzag_kv_slices_late_chunks`**
   - Tests late chunk extraction (chunks world_size to 2*world_size-1)
   - Validates correct index computation for second half of zigzag pattern

4. **`test_scatter_grad_to_zigzag_single_sequence`**
   - Validates backward gradient scattering
   - Compares Triton kernel vs Python reference

5. **`test_scatter_grad_to_zigzag_multiple_sequences`**
   - Tests gradient scattering with multiple sequences
   - Validates per-sequence offset calculations

### Performance Tests

6. **`test_triton_kernel_performance`**
   - Benchmarks Triton kernel vs Python reference
   - Validates that Triton is actually faster (speedup > 1.0x)
   - Typical speedup: 10-20x for KV extraction

## Expected Behavior

All tests should:
- ✅ Pass with **exact equality** (diff = 0.0) for correctness
- ✅ Show **significant speedup** (>5x) for performance tests
- ✅ Work with any world size (2, 4, 8 GPUs)
- ✅ Support different dtypes (float16, bfloat16, float32)

## Debugging Failed Tests

If tests fail:

1. **Check world size**: Ensure `seq_len % (2 * world_size) == 0`
2. **Check CUDA availability**: All ranks need GPU access
3. **Check Triton installation**: `pip install triton`
4. **Enable Python fallback**: Set `use_triton_kernel=False` in main code

## Integration with Main Tests

These component tests are complementary to the end-to-end test:
- **Component tests** (this directory): Test individual kernels in isolation
- **End-to-end test** (`test/test_zigzag_llama3_flash_attn_varlen_func.py`): Test full forward/backward pass

Both should pass for full validation.
