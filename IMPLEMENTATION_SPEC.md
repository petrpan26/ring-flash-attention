# Fused Varlen Kernel - Implementation Specification

## Objective

Implement a fused Triton kernel that merges two flash attention calls into one, reducing K,V memory bandwidth by 50% (from 2x to 1x reads).

## Background Context

### The Problem
The original implementation in `ring_flash_attn/zigzag_llama3_flash_attn_varlen.py` calls flash attention twice (once per Q group), reading K,V from HBM twice:

```python
# execute_grouped_attention() - INEFFICIENT
for group_idx in [0, 1]:  # Two separate calls
    out, lse = _flash_attn_varlen_forward(
        q_group, k, v, ...  # K,V read from HBM each time
    )
```

**Total K,V memory reads: 2x** (once per group)

### The Solution
Fuse both Q groups into a single kernel that loads K,V once and uses them for both groups:

```python
# Fused kernel - EFFICIENT
out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
    q0, q1, k, v, ...  # K,V read from HBM once, used for both Q groups
)
```

**Total K,V memory reads: 1x** (50% bandwidth reduction)

## Documentation References

Read these files for complete context:

1. **`FUSED_KERNEL_VARLEN_API.md`** - Complete API design with examples
2. **`FUSED_KERNEL_API_PARAMS.md`** - Detailed parameter reference
3. **`ring_flash_attn/zigzag_llama3_flash_attn_varlen.py`** - Original implementation (lines 393-526: `execute_grouped_attention`)
4. **`ring_flash_attn/triton_zigzag_llama3_flash_attn_v2.py`** - Existing V2 kernel (needs varlen support)

## API Specification

### Function Signature

```python
def fused_zigzag_llama3_flash_attn_varlen_forward(
    q0: torch.Tensor,              # [total_q0_tokens, nheads, headdim]
    q1: torch.Tensor,              # [total_q1_tokens, nheads, headdim]
    k: torch.Tensor,               # [total_k_tokens, nheads_k, headdim]
    v: torch.Tensor,               # [total_v_tokens, nheads_k, headdim]
    cu_seqlens_q0: torch.Tensor,   # [batch_size + 1], int32
    cu_seqlens_q1: torch.Tensor,   # [batch_size + 1], int32
    cu_seqlens_k: torch.Tensor,    # [batch_size + 1], int32
    max_seqlen_q0: int,
    max_seqlen_q1: int,
    max_seqlen_k: int,
    max_kv_len_q0: int,            # How far into K,V that Q0 can attend
    max_kv_len_q1: int,            # How far into K,V that Q1 can attend
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        out0: [total_q0_tokens, nheads, headdim]
        out1: [total_q1_tokens, nheads, headdim]
        lse0: [nheads, total_q0_tokens] or [batch_size, nheads, max_seqlen_q0_rounded]
        lse1: [nheads, total_q1_tokens] or [batch_size, nheads, max_seqlen_q1_rounded]
    """
```

### Key Constraints

1. **Varlen format**: All inputs are concatenated sequences
   - `q0[cu_seqlens_q0[i]:cu_seqlens_q0[i+1]]` is sequence i's Q0 tokens
   - Same for q1 and k,v

2. **Same batch size**: All three cu_seqlens arrays have `batch_size + 1` elements

3. **Different attention ranges**:
   - Q0 attends to: `k[seq_start : seq_start + max_kv_len_q0]`
   - Q1 attends to: `k[seq_start : seq_start + max_kv_len_q1]`

4. **Shared K,V**: Both Q groups use the same K,V (loaded once)

## Implementation Requirements

### File to Create

`ring_flash_attn/triton_fused_dual_group_flash_attn.py`

### Kernel Structure

```python
@triton.jit
def _fused_dual_group_fwd_kernel_varlen(
    # Q groups
    Q0, Q1,
    # Shared K,V
    K, V,
    # Outputs
    Out0, Out1, Lse0, Lse1, TMP,
    # Scalar params
    sm_scale,
    # Varlen sequence info
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_kv_len_q0, max_kv_len_q1,
    # Strides (varlen: no batch dimension)
    stride_q0h, stride_q0m,
    stride_q1h, stride_q1m,
    stride_kh, stride_kn,
    stride_vh, stride_vn,
    stride_o0h, stride_o0m,
    stride_o1h, stride_o1m,
    # Dimensions
    nheads, nheads_k,
    headdim,
    # Compile-time constants
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused forward kernel with varlen support.

    Key innovation: Load K,V once, use for both Q0 and Q1.
    """
```

### Kernel Logic (Pseudocode)

```python
# 1. Get program IDs
start_m = tl.program_id(0)  # M-block index
off_hz = tl.program_id(1)   # head * sequence
off_z = off_hz // nheads    # sequence index
off_h = off_hz % nheads     # head index

# 2. Load sequence boundaries (varlen)
seq_start_q0 = tl.load(cu_seqlens_q0 + off_z)
seq_end_q0 = tl.load(cu_seqlens_q0 + off_z + 1)
seq_len_q0 = seq_end_q0 - seq_start_q0

seq_start_q1 = tl.load(cu_seqlens_q1 + off_z)
seq_end_q1 = tl.load(cu_seqlens_q1 + off_z + 1)
seq_len_q1 = seq_end_q1 - seq_start_q1

seq_start_k = tl.load(cu_seqlens_k + off_z)
seq_end_k = tl.load(cu_seqlens_k + off_z + 1)
seq_len_k = seq_end_k - seq_start_k

# 3. Adjust pointers to this sequence
Q0 += seq_start_q0 * stride_q0m
Q1 += seq_start_q1 * stride_q1m
K += seq_start_k * stride_kn
V += seq_start_k * stride_vn

# 4. Compute attention ranges for this sequence
end_n0 = tl.minimum(max_kv_len_q0, seq_len_k)
end_n1 = tl.minimum(max_kv_len_q1, seq_len_k)
max_end_n = tl.maximum(end_n0, end_n1)

# 5. Load Q for both groups
offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
q0 = tl.load(Q0 + offs_m * stride_q0m, mask=offs_m < seq_len_q0)
q1 = tl.load(Q1 + offs_m * stride_q1m, mask=offs_m < seq_len_q1)

# Apply Dot I trick for headdim < 128
if BLOCK_HEADDIM < 128:
    I = identity_matrix()
    q0 = tl.dot(q0, I)
    q1 = tl.dot(q1, I)

# 6. Initialize accumulators for both groups
acc_o0 = zeros([BLOCK_M, BLOCK_HEADDIM])
acc_o1 = zeros([BLOCK_M, BLOCK_HEADDIM])
m_i0 = full([BLOCK_M], -inf)
m_i1 = full([BLOCK_M], -inf)
l_i0 = zeros([BLOCK_M])
l_i1 = zeros([BLOCK_M])

# 7. Loop over K,V blocks ONCE
for start_n in range(0, max_end_n, BLOCK_N):
    # Load K,V ONCE (shared by both groups)
    k = tl.load(K + start_n * stride_kn, mask=..., cache_modifier=".cg")
    v = tl.load(V + start_n * stride_vn, mask=..., cache_modifier=".cg")

    # Process Q0 (if within range)
    if start_n < end_n0:
        # Compute attention
        qk0 = tl.dot(q0, k, trans_b=True)

        # Apply masks (causal + range)
        mask0 = create_mask(offs_m, start_n, end_n0, IS_CAUSAL)
        qk0 = tl.where(mask0, qk0 * sm_scale * log2e, -inf)

        # Online softmax
        m_i_new0 = tl.maximum(m_i0, tl.max(qk0, 1))
        alpha0 = tl.math.exp2((m_i0 - m_i_new0) * log2e)
        p0 = tl.math.exp2(qk0 - m_i_new0[:, None])

        # Update accumulator
        acc_o0 = acc_o0 * alpha0[:, None] + tl.dot(p0, v)
        l_i0 = l_i0 * alpha0 + tl.sum(p0, 1)
        m_i0 = m_i_new0

    # Process Q1 (if within range) - REUSING same k, v!
    if start_n < end_n1:
        # Same computation as Q0, but with Q1's data
        qk1 = tl.dot(q1, k, trans_b=True)
        mask1 = create_mask(offs_m, start_n, end_n1, IS_CAUSAL)
        qk1 = tl.where(mask1, qk1 * sm_scale * log2e, -inf)

        m_i_new1 = tl.maximum(m_i1, tl.max(qk1, 1))
        alpha1 = tl.math.exp2((m_i1 - m_i_new1) * log2e)
        p1 = tl.math.exp2(qk1 - m_i_new1[:, None])

        acc_o1 = acc_o1 * alpha1[:, None] + tl.dot(p1, v)  # Same v!
        l_i1 = l_i1 * alpha1 + tl.sum(p1, 1)
        m_i1 = m_i_new1

# 8. Final scaling and write outputs
acc_o0 = acc_o0 / l_i0[:, None]
lse0 = m_i0 / log2e + tl.log(l_i0)
tl.store(Out0 + seq_start_q0 + offs_m * stride_o0m, acc_o0, mask=...)
tl.store(Lse0 + ..., lse0, mask=...)

acc_o1 = acc_o1 / l_i1[:, None]
lse1 = m_i1 / log2e + tl.log(l_i1)
tl.store(Out1 + seq_start_q1 + offs_m * stride_o1m, acc_o1, mask=...)
tl.store(Lse1 + ..., lse1, mask=...)
```

### Python Wrapper

```python
def fused_zigzag_llama3_flash_attn_varlen_forward(
    q0, q1, k, v,
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
    max_kv_len_q0, max_kv_len_q1,
    softmax_scale=None,
    causal=True,
):
    # Validate inputs
    batch_size = len(cu_seqlens_q0) - 1
    assert len(cu_seqlens_q1) == batch_size + 1
    assert len(cu_seqlens_k) == batch_size + 1

    nheads = q0.shape[1]
    nheads_k = k.shape[1]
    headdim = q0.shape[2]

    # Allocate outputs
    out0 = torch.empty_like(q0)
    out1 = torch.empty_like(q1)

    # LSE tensors
    lse0 = torch.empty((batch_size, nheads, max_seqlen_q0_rounded), ...)
    lse1 = torch.empty((batch_size, nheads, max_seqlen_q1_rounded), ...)

    # Get hardware-specific config
    config = get_fused_fwd_config(batch_size, nheads, max_seqlen_q0, max_seqlen_q1, max_seqlen_k, headdim, causal)
    BLOCK_M, BLOCK_N, num_stages, num_warps = config

    # Launch kernel
    grid = (triton.cdiv(max(max_seqlen_q0, max_seqlen_q1), BLOCK_M), batch_size * nheads)

    _fused_dual_group_fwd_kernel_varlen[grid](
        q0, q1, k, v,
        out0, out1, lse0, lse1, tmp,
        softmax_scale,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        max_kv_len_q0, max_kv_len_q1,
        ...,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out0, out1, lse0, lse1
```

### Integration Wrapper

Create a drop-in replacement for `execute_grouped_attention`:

```python
def replace_execute_grouped_attention_with_fused_varlen(
    chunk_q_list,
    chunk_cu_seqlens_q_list,
    chunk_indices_list,
    kv_buffer,
    kv_slices,
    nheads,
    head_dim,
    softmax_scale,
    dropout_p,
    causal,
    window_size,
    alibi_slopes,
    deterministic,
    world_size,
    cu_seqlens_k,
    chunk_idx_0,
    chunk_idx_1,
):
    """Drop-in replacement for execute_grouped_attention."""
    # Rearrange KV to contiguous
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size, cu_seqlens_k)
    k = kv_contiguous[0]
    v = kv_contiguous[1]

    # Extract Q groups
    q0 = chunk_q_list[0]
    q1 = chunk_q_list[1]
    cu_seqlens_q0 = chunk_cu_seqlens_q_list[0]
    cu_seqlens_q1 = chunk_cu_seqlens_q_list[1]

    # Compute max_seqlen
    max_seqlen_q0 = (cu_seqlens_q0[1:] - cu_seqlens_q0[:-1]).max().item()
    max_seqlen_q1 = (cu_seqlens_q1[1:] - cu_seqlens_q1[:-1]).max().item()
    cu_seqlens_k_global = cu_seqlens_k * world_size
    max_seqlen_k = (cu_seqlens_k_global[1:] - cu_seqlens_k_global[:-1]).max().item()

    # Compute attention ranges from chunk indices
    total_chunks = 2 * world_size
    chunk_size = max_seqlen_k // total_chunks
    max_kv_len_q0 = (chunk_idx_0 + 1) * chunk_size
    max_kv_len_q1 = (chunk_idx_1 + 1) * chunk_size

    # Call fused kernel
    out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k_global,
        max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
        max_kv_len_q0, max_kv_len_q1,
        softmax_scale, causal,
    )

    # Reconstruct output (scatter back to original positions)
    total_q = sum(q.shape[0] for q in chunk_q_list)
    out = torch.zeros((total_q, nheads, head_dim), dtype=out0.dtype, device=out0.device)
    out[chunk_indices_list[0]] = out0
    out[chunk_indices_list[1]] = out1

    # Reconstruct LSE
    lse = torch.zeros((nheads, total_q), dtype=lse0.dtype, device=lse0.device)
    lse[:, chunk_indices_list[0]] = lse0[:, :len(chunk_indices_list[0])]
    lse[:, chunk_indices_list[1]] = lse1[:, :len(chunk_indices_list[1])]

    chunk_info = {
        'chunk_indices_list': chunk_indices_list,
        'chunk_cu_seqlens_q_list': chunk_cu_seqlens_q_list,
        'kv_slices': kv_slices,
    }

    return out, lse, chunk_info
```

## Test Requirements

### File to Create

`test/test_fused_dual_group_flash_attn_varlen.py`

### Test Cases

#### Test 1: Correctness - Single Sequence
```python
def test_single_sequence_correctness():
    """Test with batch_size=1, verify outputs match original."""
    batch_size = 1
    seqlen_local = 256
    nheads = 8
    headdim = 64
    world_size = 4
    rank = 0

    # Create inputs
    q = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device='cuda')
    k = torch.randn(seqlen_local * world_size, nheads, headdim, dtype=torch.float16, device='cuda')
    v = torch.randn(seqlen_local * world_size, nheads, headdim, dtype=torch.float16, device='cuda')

    # Split Q into groups
    q0 = q[:seqlen_local // 2]
    q1 = q[seqlen_local // 2:]

    cu_seqlens_q0 = torch.tensor([0, seqlen_local // 2], dtype=torch.int32, device='cuda')
    cu_seqlens_q1 = torch.tensor([0, seqlen_local // 2], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, seqlen_local * world_size], dtype=torch.int32, device='cuda')

    # Compute attention ranges
    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    total_chunks = 2 * world_size
    chunk_size = (seqlen_local * world_size) // total_chunks
    max_kv_len_q0 = (chunk_idx_0 + 1) * chunk_size
    max_kv_len_q1 = (chunk_idx_1 + 1) * chunk_size

    # Run fused kernel
    out0_fused, out1_fused, lse0_fused, lse1_fused = fused_zigzag_llama3_flash_attn_varlen_forward(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        seqlen_local // 2, seqlen_local // 2, seqlen_local * world_size,
        max_kv_len_q0, max_kv_len_q1,
        causal=True,
    )

    # Run original (two separate calls)
    out0_orig, lse0_orig = _flash_attn_varlen_forward(
        q0, k[:max_kv_len_q0], v[:max_kv_len_q0],
        cu_seqlens_q0, torch.tensor([0, max_kv_len_q0], dtype=torch.int32, device='cuda'),
        seqlen_local // 2, max_kv_len_q0,
        causal=True,
    )

    out1_orig, lse1_orig = _flash_attn_varlen_forward(
        q1, k[:max_kv_len_q1], v[:max_kv_len_q1],
        cu_seqlens_q1, torch.tensor([0, max_kv_len_q1], dtype=torch.int32, device='cuda'),
        seqlen_local // 2, max_kv_len_q1,
        causal=True,
    )

    # Compare
    assert torch.allclose(out0_fused, out0_orig, atol=1e-2, rtol=1e-2), "Q0 outputs don't match"
    assert torch.allclose(out1_fused, out1_orig, atol=1e-2, rtol=1e-2), "Q1 outputs don't match"
    print("✓ Single sequence test passed")
```

#### Test 2: Correctness - Multiple Sequences (Varlen)
```python
def test_multi_sequence_varlen_correctness():
    """Test with batch_size=3, different sequence lengths."""
    batch_size = 3
    seq_lens_q0 = [30, 80, 50]  # Different Q0 lengths
    seq_lens_q1 = [70, 120, 90]  # Different Q1 lengths
    seq_lens_k = [400, 800, 600]  # Different K,V lengths

    # Create varlen inputs
    total_q0_tokens = sum(seq_lens_q0)
    total_q1_tokens = sum(seq_lens_q1)
    total_k_tokens = sum(seq_lens_k)

    q0 = torch.randn(total_q0_tokens, 8, 64, dtype=torch.float16, device='cuda')
    q1 = torch.randn(total_q1_tokens, 8, 64, dtype=torch.float16, device='cuda')
    k = torch.randn(total_k_tokens, 8, 64, dtype=torch.float16, device='cuda')
    v = torch.randn(total_k_tokens, 8, 64, dtype=torch.float16, device='cuda')

    # cu_seqlens
    cu_seqlens_q0 = torch.tensor([0, 30, 110, 160], dtype=torch.int32, device='cuda')
    cu_seqlens_q1 = torch.tensor([0, 70, 190, 280], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, 400, 1200, 1800], dtype=torch.int32, device='cuda')

    # Test with different max_kv_len for each group
    max_kv_len_q0 = 100
    max_kv_len_q1 = 800

    # Run fused kernel
    out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
        q0, q1, k, v,
        cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
        max(seq_lens_q0), max(seq_lens_q1), max(seq_lens_k),
        max_kv_len_q0, max_kv_len_q1,
        causal=True,
    )

    # Verify per-sequence by calling original on each sequence
    for i in range(batch_size):
        q0_seq = q0[cu_seqlens_q0[i]:cu_seqlens_q0[i+1]]
        q1_seq = q1[cu_seqlens_q1[i]:cu_seqlens_q1[i+1]]
        k_seq = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
        v_seq = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]]

        # Test Q0
        k0_range = min(max_kv_len_q0, seq_lens_k[i])
        out0_expected, _ = _flash_attn_varlen_forward(
            q0_seq, k_seq[:k0_range], v_seq[:k0_range],
            torch.tensor([0, seq_lens_q0[i]], dtype=torch.int32, device='cuda'),
            torch.tensor([0, k0_range], dtype=torch.int32, device='cuda'),
            seq_lens_q0[i], k0_range, causal=True,
        )

        out0_actual = out0[cu_seqlens_q0[i]:cu_seqlens_q0[i+1]]
        assert torch.allclose(out0_actual, out0_expected, atol=1e-2, rtol=1e-2), f"Q0 seq {i} mismatch"

    print("✓ Multi-sequence varlen test passed")
```

#### Test 3: Integration Test
```python
def test_integration_with_execute_grouped_attention():
    """Test drop-in replacement for execute_grouped_attention."""
    from ring_flash_attn.zigzag_llama3_flash_attn_varlen import (
        split_q_by_zigzag_chunk_index,
        compute_kv_slices_for_groups,
        execute_grouped_attention,
        rearrange_kv_from_zigzag_to_contiguous,
    )

    world_size = 4
    rank = 0
    seqlen_local = 256
    nheads = 8
    headdim = 64

    # Create test data
    q = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device='cuda')
    k = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device='cuda')
    v = torch.randn(seqlen_local, nheads, headdim, dtype=torch.float16, device='cuda')

    kv_buffer = torch.stack([
        k.repeat(world_size, 1, 1),
        v.repeat(world_size, 1, 1)
    ])

    cu_seqlens_q = torch.tensor([0, seqlen_local], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, seqlen_local], dtype=torch.int32, device='cuda')

    # Prepare inputs
    chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q, cu_seqlens_q, world_size, rank, n_chunks=2
    )

    chunk_idx_0 = rank
    chunk_idx_1 = 2 * world_size - 1 - rank
    kv_slices = compute_kv_slices_for_groups(
        cu_seqlens_k, chunk_idx_0, chunk_idx_1, world_size,
        chunk_cu_seqlens_q_list, n_chunks=2
    )

    softmax_scale = 1.0 / (headdim ** 0.5)

    # Run original
    out_orig, lse_orig, _ = execute_grouped_attention(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k
    )

    # Run fused
    out_fused, lse_fused, _ = replace_execute_grouped_attention_with_fused_varlen(
        chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list,
        kv_buffer, kv_slices,
        nheads, headdim, softmax_scale,
        dropout_p=0.0, causal=True,
        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
        world_size=world_size, cu_seqlens_k=cu_seqlens_k,
        chunk_idx_0=chunk_idx_0, chunk_idx_1=chunk_idx_1
    )

    # Compare
    assert torch.allclose(out_fused, out_orig, atol=1e-2, rtol=1e-2), "Outputs don't match"
    print("✓ Integration test passed")
```

#### Test 4: Performance Benchmark
```python
def test_performance_comparison():
    """Benchmark fused vs original."""
    import time

    configs = [
        (256, 8, 64),
        (512, 16, 64),
        (512, 32, 128),
    ]

    world_size = 4
    num_iters = 50

    for seqlen, nheads, headdim in configs:
        # Setup
        q = torch.randn(seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
        k = torch.randn(seqlen * world_size, nheads, headdim, dtype=torch.float16, device='cuda')
        v = torch.randn(seqlen * world_size, nheads, headdim, dtype=torch.float16, device='cuda')

        # ... (same setup as integration test)

        # Benchmark original
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            out_orig, _, _ = execute_grouped_attention(...)
        torch.cuda.synchronize()
        time_orig = (time.time() - start) / num_iters

        # Benchmark fused
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            out_fused, _, _ = replace_execute_grouped_attention_with_fused_varlen(...)
        torch.cuda.synchronize()
        time_fused = (time.time() - start) / num_iters

        speedup = time_orig / time_fused
        print(f"seqlen={seqlen}, nheads={nheads}, headdim={headdim}: {speedup:.2f}x speedup")
        assert speedup > 1.3, f"Expected >1.3x speedup, got {speedup:.2f}x"
```

## Optimization Checklist

### FlagAttention Optimizations to Include

1. **log2e/exp2**: Use `tl.math.exp2(x * log2e)` instead of `tl.exp(x)`
2. **Cache modifiers**: Use `cache_modifier=".cg"` on K,V loads
3. **Dot I trick**: For headdim < 128, apply identity matrix multiplication
4. **Hardware configs**: Use device-specific BLOCK_M, BLOCK_N, num_warps, num_stages
5. **GQA support**: Handle nheads != nheads_k correctly

### Memory Access Pattern

```
Load K,V block once → Use for Q0 → Use for Q1 (same block!)
    ↓                    ↓              ↓
  HBM read          SRAM reuse     SRAM reuse
```

## Success Criteria

1. **Correctness**: All tests pass with tolerance atol=1e-2, rtol=1e-2
2. **Performance**: Achieve >1.5x speedup over original
3. **Memory**: Verify K,V read only once (profile with nsys if possible)
4. **API**: Drop-in replacement for `execute_grouped_attention`
5. **Varlen**: Works with different sequence lengths per batch item

## Expected Results

- **Forward speedup**: 1.5-2.0x
- **Memory bandwidth**: 50% reduction (2x → 1x for K,V)
- **Numerical match**: Within fp16 tolerance of original

## Additional Notes

- Start with forward pass only (backward can be added later)
- Use existing V2 kernel as reference for optimizations
- Focus on correctness first, then optimize
- Test on different hardware if available (A100, H100, RTX-3090)
