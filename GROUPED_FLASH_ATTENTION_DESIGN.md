# Grouped Flash Attention - Implementation Design

## Executive Summary

This document outlines the design for adding **grouped attention** support to flash-attention, enabling multiple Q groups to share K,V loads in a single kernel invocation. This optimization will eliminate redundant HBM reads in zigzag_llama3, providing an estimated **15-20% speedup** in two-kernels mode.

**Complexity Assessment:** High - requires modifying flash-attention's CUDA kernels
**Estimated Time:** 2-3 weeks for experienced CUDA developer
**Recommended Approach:** Start with Python API prototype, then implement CUDA kernel changes

---

## Problem Statement

### Current Implementation (Two Separate Kernel Calls)

In `ring_flash_attn/zigzag_llama3_flash_attn_varlen.py:612`, we call flash attention twice:

```python
# Call 1: Early group
out_early, lse_early, _, _ = _flash_attn_varlen_forward(
    q=q_early,           # [tokens_early, nheads, dim]
    k=k[:tokens_early],  # [tokens_early, nheads_k, dim]
    v=v[:tokens_early],  # Loads K,V[:tokens_early] from HBM
    ...
)

# Call 2: Late group
out_late, lse_late, _, _ = _flash_attn_varlen_forward(
    q=q_late,            # [tokens_late, nheads, dim]
    k=k[:tokens_late],   # [tokens_late, nheads_k, dim]
    v=v[:tokens_late],   # Loads K,V[:tokens_early] AGAIN + K,V[tokens_early:tokens_late]
    ...
)
```

### Redundant Memory Access

The overlapping region `k,v[:tokens_early]` is loaded from HBM **twice**:
- **Early call**: Loads full `k,v[:tokens_early]` (~134 MB for Llama3-8B @ 65K tokens)
- **Late call**: Loads full `k,v[:tokens_late]` (~268 MB), which includes `k,v[:tokens_early]` again

**Total HBM reads**: 134 + 268 = **402 MB**
**Optimal HBM reads**: 268 MB (if shared)
**Waste**: 134 MB (~33% overhead)

---

## Proposed Solution

### New API: `_flash_attn_varlen_forward_grouped`

Add a new function to `flash_attn/flash_attn_interface.py`:

```python
def _flash_attn_varlen_forward_grouped(
    q_list: List[torch.Tensor],              # List of Q tensors, one per group
    k: torch.Tensor,                         # Shared K tensor (full length)
    v: torch.Tensor,                         # Shared V tensor (full length)
    cu_seqlens_q_list: List[torch.Tensor],   # List of cu_seqlens_q per group
    cu_seqlens_k_list: List[torch.Tensor],   # List of cu_seqlens_k with different end positions
    max_seqlen_q_list: List[int],            # Max seqlen_q per group
    max_seqlen_k_list: List[int],            # Max seqlen_k per group (DIFFERENT!)
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Flash attention with multiple Q groups sharing K,V loads.

    Each Q group attends to a different-length prefix of K,V.
    K,V are loaded only once per tile, shared across all groups.

    Args:
        q_list: List of Q tensors [q_group0, q_group1, ...]
        k, v: Shared K,V tensors (full sequence length)
        cu_seqlens_k_list: List of cu_seqlens_k with different endpoints
                           (e.g., [cu_seqlens_k_early, cu_seqlens_k_late])
        max_seqlen_k_list: Max K sequence lengths per group
                           (e.g., [tokens_early, tokens_late])
        ... (other params same as _flash_attn_varlen_forward)

    Returns:
        out_list: List of output tensors, one per group
        lse_list: List of LSE tensors, one per group
        S_dmask: Combined dropout mask (if return_softmax)
        rng_state: RNG state

    Example:
        # Zigzag llama3 use case (2 groups):
        out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
            q_list=[q_early, q_late],
            k=kv_contiguous[0],  # Full K
            v=kv_contiguous[1],  # Full V
            cu_seqlens_q_list=[cu_seqlens_q_early, cu_seqlens_q_late],
            cu_seqlens_k_list=[cu_seqlens_k_early, cu_seqlens_k_late],
            max_seqlen_q_list=[max_seqlen_q, max_seqlen_q],
            max_seqlen_k_list=[tokens_early, tokens_late],  # Different!
            ...
        )
    """
    pass
```

### Key Design Decisions

#### 1. Multiple Q Groups, Single K,V
- **Input**: List of Q tensors, but single K,V tensor
- **Rationale**: K,V are shared; only Q is split into groups

#### 2. Group-Specific K,V Slice Lengths
- **Critical feature**: Each group can attend to different K,V prefix lengths
- **Enabled by**: `cu_seqlens_k_list` and `max_seqlen_k_list` parameters
- **Example**:
  - Group 0 (early): attends to K,V[:4096]
  - Group 1 (late): attends to K,V[:8192]

#### 3. Separate Outputs per Group
- **Returns**: Lists of outputs and LSE, one per group
- **Rationale**: Each group has independent attention computation

---

## Implementation Strategy

### Phase 1: Python API Prototype (Week 1)

**Goal**: Validate API design and measure potential benefits

**Approach**: Create Python wrapper that launches existing kernels strategically

```python
# In flash_attn/flash_attn_interface.py

def _flash_attn_varlen_forward_grouped(
    q_list, k, v, cu_seqlens_q_list, cu_seqlens_k_list,
    max_seqlen_q_list, max_seqlen_k_list, ...
):
    """Python prototype using existing kernels."""

    out_list = []
    lse_list = []

    # Launch kernels on same stream
    for i, q_group in enumerate(q_list):
        k_end = cu_seqlens_k_list[i][-1].item()

        # Use existing _flash_attn_varlen_forward
        out, lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q=q_group,
            k=k[:k_end],  # Slice K,V to group-specific length
            v=v[:k_end],
            cu_seqlens_q=cu_seqlens_q_list[i],
            cu_seqlens_k=cu_seqlens_k_list[i],
            max_seqlen_q=max_seqlen_q_list[i],
            max_seqlen_k=max_seqlen_k_list[i],
            ...
        )

        out_list.append(out)
        lse_list.append(lse)

    return out_list, lse_list, S_dmask, rng_state
```

**Benefits**:
- Validates API design
- Tests integration with zigzag_llama3
- Measures L2 cache hit rates (may provide 5-10% speedup from caching)
- **No CUDA changes needed**

**Deliverables**:
- Python API implementation
- Integration test with zigzag_llama3
- Performance profiling (Nsight Systems)

---

### Phase 2: CUDA Kernel Modification (Weeks 2-3)

**Goal**: Implement grouped attention in CUDA for maximum performance

#### Step 2.1: Extend Flash_fwd_params Struct

Modify `csrc/flash_attn/src/flash.h`:

```cpp
struct Flash_fwd_params : public Qkv_params {
    // ... existing fields ...

    // NEW: Grouped attention support
    int num_groups;                          // Number of Q groups (e.g., 2)
    int* group_q_offsets;                    // Cumulative Q token offsets per group [g0_start, g0_end, g1_start, g1_end, ...]
    int* group_max_seqlen_k;                 // Max K,V length per group [tokens_early, tokens_late]
    int** group_cu_seqlens_k;                // Array of cu_seqlens_k pointers, one per group

    // Output pointers per group
    void** group_o_ptrs;                     // Array of output pointers
    void** group_softmax_lse_ptrs;           // Array of LSE pointers
};
```

#### Step 2.2: Modify Block Assignment Logic

In `csrc/flash_attn/src/flash_fwd_kernel.h`, modify kernel launch to handle groups:

**Current grid dimensions**:
```cpp
dim3 grid(num_q_blocks, num_heads, batch_size);
```

**Proposed grouped grid**:
```cpp
// Total Q blocks across all groups
int total_q_blocks = 0;
for (int g = 0; g < num_groups; g++) {
    total_q_blocks += group_q_blocks[g];
}

dim3 grid(total_q_blocks, num_heads, batch_size);
```

**Kernel modification**:
```cpp
template<...>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
    int block_id = blockIdx.x;
    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;

    // NEW: Determine which group this block belongs to
    int group_id = 0;
    int local_block_id = block_id;

    for (int g = 0; g < params.num_groups; g++) {
        int group_num_blocks = params.group_q_offsets[g * 2 + 1] - params.group_q_offsets[g * 2];
        group_num_blocks = (group_num_blocks + kBlockM - 1) / kBlockM;

        if (local_block_id < group_num_blocks) {
            group_id = g;
            break;
        }
        local_block_id -= group_num_blocks;
    }

    // Load Q block for this group
    int q_offset = params.group_q_offsets[group_id * 2] + local_block_id * kBlockM;
    // ... existing Q loading logic ...

    // Loop over K,V blocks (up to group-specific limit)
    int max_kv_len = params.group_max_seqlen_k[group_id];
    int n_block_max = (max_kv_len + kBlockN - 1) / kBlockN;

    // Use group-specific cu_seqlens_k for varlen support
    int* cu_seqlens_k_group = params.group_cu_seqlens_k[group_id];

    for (int kv_block_id = 0; kv_block_id < n_block_max; kv_block_id++) {
        // Load K,V tile (SHARED across groups if in overlapping region!)
        // ... existing K,V loading logic ...

        // Compute attention
        // ... existing attention computation ...
    }

    // Write output to group-specific buffer
    void* output_ptr = params.group_o_ptrs[group_id];
    void* lse_ptr = params.group_softmax_lse_ptrs[group_id];
    // ... write output ...
}
```

#### Step 2.3: K,V Tile Sharing Optimization

**Critical optimization**: Ensure K,V tiles are loaded only once even when used by multiple groups.

**Challenge**: Different thread blocks (from different groups) may need the same K,V tile at different times.

**Solution**: Rely on L2 cache for automatic sharing
- K,V tiles loaded by early group remain in L2 cache
- Late group reuses from L2 (cache hit) instead of HBM (cache miss)
- Modern GPUs (A100/H100) have large L2 caches (40-50 MB)

**Expected L2 cache hit rate**: ~80-90% for overlapping K,V regions

**Alternative (advanced)**: Explicit shared memory coordination
- Use inter-block communication (requires CUDA Dynamic Parallelism or cooperative groups)
- Much more complex; not recommended for first implementation

#### Step 2.4: Update Python Interface

Modify `csrc/flash_attn/flash_api.cpp`:

```cpp
// Add new function
std::vector<at::Tensor>
mha_varlen_fwd_grouped(
    std::vector<at::Tensor> &q_list,        // List of Q tensors
    const at::Tensor &k,                     // Shared K
    const at::Tensor &v,                     // Shared V
    const std::vector<at::Tensor> &cu_seqlens_q_list,
    const std::vector<at::Tensor> &cu_seqlens_k_list,
    const std::vector<int> &max_seqlen_q_list,
    const std::vector<int> &max_seqlen_k_list,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_
) {
    // ... validation checks ...

    // Allocate outputs for each group
    std::vector<at::Tensor> out_list;
    std::vector<at::Tensor> lse_list;

    for (size_t i = 0; i < q_list.size(); i++) {
        out_list.push_back(torch::empty_like(q_list[i]));
        lse_list.push_back(torch::empty({...}, opts.dtype(at::kFloat)));
    }

    // Setup grouped params
    Flash_fwd_params params;
    params.num_groups = q_list.size();

    // Allocate and populate group metadata on GPU
    // ... setup group_q_offsets, group_max_seqlen_k, etc. ...

    // Launch grouped kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd_grouped(params, stream);

    // Return results
    std::vector<at::Tensor> results;
    for (size_t i = 0; i < q_list.size(); i++) {
        results.push_back(out_list[i]);
        results.push_back(lse_list[i]);
    }
    return results;
}

// Update PYBIND11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ... existing bindings ...
    m.def("varlen_fwd_grouped", &FLASH_NAMESPACE::mha_varlen_fwd_grouped,
          "Forward pass (variable length, grouped)");
}
```

---

### Phase 3: Testing and Validation

#### Unit Tests

```python
# test/test_flash_attn_grouped.py

def test_grouped_forward_correctness():
    """Test that grouped attention produces same results as separate calls."""

    # Setup
    q_early = torch.randn(1000, 32, 128, device='cuda', dtype=torch.float16)
    q_late = torch.randn(1000, 32, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(2000, 8, 128, device='cuda', dtype=torch.float16)
    v = torch.randn(2000, 8, 128, device='cuda', dtype=torch.float16)

    # Reference: separate calls
    out_early_ref, lse_early_ref, _, _ = _flash_attn_varlen_forward(
        q_early, k[:1000], v[:1000], ...
    )
    out_late_ref, lse_late_ref, _, _ = _flash_attn_varlen_forward(
        q_late, k[:2000], v[:2000], ...
    )

    # Test: grouped call
    out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
        [q_early, q_late], k, v, ...,
        max_seqlen_k_list=[1000, 2000]
    )

    # Verify
    assert torch.allclose(out_list[0], out_early_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out_list[1], out_late_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(lse_list[0], lse_early_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(lse_list[1], lse_late_ref, rtol=1e-3, atol=1e-3)
```

#### Integration Test

```python
# test/test_zigzag_llama3_grouped.py

def test_zigzag_llama3_with_grouped_attention():
    """Test zigzag_llama3 using grouped attention."""

    # Run with two-kernels mode (original)
    out_ref, lse_ref = zigzag_llama3_flash_attn_varlen_func(
        ..., use_fused_kernel_forward=False, use_grouped_attention=False
    )

    # Run with grouped attention
    out_test, lse_test = zigzag_llama3_flash_attn_varlen_func(
        ..., use_fused_kernel_forward=False, use_grouped_attention=True
    )

    # Verify correctness
    assert torch.allclose(out_test, out_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(lse_test, lse_ref, rtol=1e-3, atol=1e-3)
```

#### Performance Benchmarks

```python
# benchmark/benchmark_grouped_attention.py

@triton.testing.perf_report(...)
def benchmark_grouped_vs_separate(tokens_early, tokens_late, provider):
    """Benchmark grouped attention vs separate kernel calls."""

    if provider == 'separate':
        # Two separate calls (current implementation)
        torch.cuda.synchronize()
        start = time.time()
        out1, lse1, _, _ = _flash_attn_varlen_forward(q1, k[:tokens_early], v[:tokens_early], ...)
        out2, lse2, _, _ = _flash_attn_varlen_forward(q2, k[:tokens_late], v[:tokens_late], ...)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    elif provider == 'grouped':
        # Single grouped call
        torch.cuda.synchronize()
        start = time.time()
        out_list, lse_list, _, _ = _flash_attn_varlen_forward_grouped(
            [q1, q2], k, v, ..., max_seqlen_k_list=[tokens_early, tokens_late]
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start

    return elapsed
```

---

## Expected Performance Gains

### Memory Bandwidth Savings

**Llama3-8B @ 65K tokens (world_size=8)**:

| Mode | K,V HBM Reads | Speedup vs Baseline |
|------|---------------|---------------------|
| **Separate calls (baseline)** | 402 MB | 1.0× |
| **Grouped (Python, L2 cache)** | ~320 MB | 1.25× (5-10% faster) |
| **Grouped (CUDA optimized)** | 268 MB | 1.50× (15-20% faster) |

### End-to-End Impact

For zigzag_llama3 forward pass:
- Current efficiency: 59.1% (two-kernels baseline)
- **Target efficiency: ~70-75%** (with grouped attention)
- **Overall speedup: +18-25%** compared to current two-kernels mode

**Note**: These are theoretical estimates. Actual gains depend on:
- L2 cache hit rates
- Memory bandwidth saturation
- Kernel launch overhead
- Overlap with computation

---

## Risks and Mitigations

### Risk 1: CUDA Complexity
**Impact**: Development takes longer than estimated
**Probability**: High
**Mitigation**: Start with Phase 1 (Python prototype) to validate approach

### Risk 2: Limited L2 Cache Benefit
**Impact**: Performance gains less than expected (5-10% instead of 15-20%)
**Probability**: Medium
**Mitigation**: Profile with Nsight Systems to measure actual cache hit rates

### Risk 3: Kernel Launch Overhead
**Impact**: Multiple groups add overhead that negates memory savings
**Probability**: Low
**Mitigation**: Use persistent kernels or fused launch to minimize overhead

### Risk 4: Maintenance Burden
**Impact**: Hard to keep up with upstream flash-attention changes
**Probability**: Medium
**Mitigation**:
- Submit changes as PR to flash-attention upstream
- Keep fork minimal and well-documented

---

## Alternative Approaches

### Alternative 1: Triton Custom Kernel

**Pros**:
- Easier to implement than CUDA (1 week vs 2-3 weeks)
- More maintainable
- Can achieve 80-90% of CUDA performance

**Cons**:
- Slower than hand-tuned CUDA
- May not match flash-attention's optimizations

**Recommendation**: Consider if CUDA approach proves too complex

### Alternative 2: Wait for Flash Attention 3

**Pros**:
- No implementation effort
- Official support

**Cons**:
- Unknown timeline (could be 3-6+ months)
- No guarantee grouped attention will be included

**Recommendation**: Only if not time-sensitive

### Alternative 3: Use Fused Mode Instead

**Pros**:
- Already implemented in zigzag_llama3
- Provides ~5% speedup over two-kernels

**Cons**:
- Uses 2× memory (duplicates K,V)
- Less efficient than grouped approach

**Recommendation**: Fallback if grouped implementation blocked

---

## Recommended Path Forward

### Option A: Full Implementation (Ambitious)

**Timeline**: 3 weeks
**Effort**: High
**Reward**: 15-20% speedup, publishable optimization

1. **Week 1**: Python prototype + profiling
2. **Week 2**: CUDA kernel modifications
3. **Week 3**: Testing, optimization, integration

**Best for**: Research project, publishable work, long-term optimization

### Option B: Python Prototype Only (Pragmatic)

**Timeline**: 3-5 days
**Effort**: Low
**Reward**: 5-10% speedup, validates approach

1. Implement Python API wrapper
2. Integrate with zigzag_llama3
3. Profile and measure L2 cache benefits
4. Decide if CUDA implementation warranted

**Best for**: Quick win, de-risking before major investment

### Option C: Triton Implementation (Middle Ground)

**Timeline**: 1 week
**Effort**: Medium
**Reward**: 10-15% speedup, maintainable

1. Write Triton fused grouped attention kernel
2. Integrate with Python API
3. Test and optimize

**Best for**: Balance of performance and maintainability

---

## Recommendation

**Start with Option B (Python Prototype)**, then decide:

1. **If L2 cache provides 8-10% speedup** → Ship it! May be good enough.
2. **If L2 cache provides <5% speedup** → Proceed to Option A (CUDA) or Option C (Triton)
3. **If complexity too high** → Fall back to fused mode (already implemented)

**Why this approach**:
- De-risks the project
- Validates API design
- Provides quick wins
- Informs decision on full CUDA implementation

---

## Next Steps

1. ✅ Create this design document
2. **[YOU ARE HERE]** Review and approve approach
3. Implement Python prototype (3-5 days)
4. Profile and measure benefits
5. Decision point: Continue to CUDA or ship Python version

---

## References

- Flash Attention paper: https://arxiv.org/abs/2205.14135
- Flash Attention 2 paper: https://arxiv.org/abs/2307.08691
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Triton documentation: https://triton-lang.org/
- Our implementation: `ring_flash_attn/zigzag_llama3_flash_attn_varlen.py`
