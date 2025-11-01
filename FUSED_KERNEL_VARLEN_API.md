# Fused Kernel API Design for Varlen Support

## Problem Statement

We want to merge two flash attention kernel calls into one to reduce K,V memory reads from 2x to 1x:

```python
# BEFORE: Two separate calls (inefficient - reads K,V twice)
out0 = flash_attn(q0, k, v, cu_seqlens_q0, cu_seqlens_k, ...)  # Read K,V #1
out1 = flash_attn(q1, k, v, cu_seqlens_q1, cu_seqlens_k, ...)  # Read K,V #2

# AFTER: Single fused call (efficient - reads K,V once)
out0, out1 = fused_flash_attn(q0, q1, k, v, cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k, ...)
```

## Key Constraint

- **Two different Q sequences** (Q0 and Q1)
- **Single shared K,V**
- Both are varlen (variable-length sequences)
- Both Q groups come from the **same original sequences**, just split differently
- **Different attention ranges**: Q0 and Q1 attend to different portions of K,V based on chunk indices

## Varlen Format Explanation

### What is Varlen?

Varlen allows batching multiple sequences of different lengths efficiently without padding:

```python
# Example: Batch of 3 sequences with different lengths
sequences = [
    [tok1, tok2],              # sequence 0: length 2
    [tok3, tok4, tok5, tok6],  # sequence 1: length 4
    [tok7]                     # sequence 2: length 1
]

# Varlen format: concatenate all sequences
q_varlen = [tok1, tok2, tok3, tok4, tok5, tok6, tok7]  # shape: [7, nheads, headdim]

# Track boundaries with cumulative sequence lengths
cu_seqlens_q = [0, 2, 6, 7]  # shape: [batch_size + 1]
# Sequence 0: q_varlen[0:2]   (length 2)
# Sequence 1: q_varlen[2:6]   (length 4)
# Sequence 2: q_varlen[6:7]   (length 1)
```

### Why Varlen?

- **No padding waste**: Only store actual tokens, not padding
- **Better memory efficiency**: Especially when sequence lengths vary significantly
- **Standard format**: Used by flash-attention library

## Our Use Case: Zigzag Distribution

In zigzag ring attention, each rank's local Q is **split into 2 groups** based on chunk indices:

```python
# Original local Q for this rank
q_local = [seq0_tokens, seq1_tokens, seq2_tokens, ...]  # [total_local_tokens, nheads, headdim]

# Split into 2 groups by zigzag pattern
q0 = [seq0_early, seq1_early, seq2_early, ...]  # Group 0: early chunks
q1 = [seq0_late, seq1_late, seq2_late, ...]     # Group 1: late chunks

# After all-gather, we have full K,V
k = [seq0_all, seq1_all, seq2_all, ...]  # Full sequences after all-gather
v = [seq0_all, seq1_all, seq2_all, ...]
```

### Critical Insight

**Each sequence is split into TWO parts:**
- Part in Q0 (early chunk indices)
- Part in Q1 (late chunk indices)

**The batch size is the same across Q0, Q1, and K,V** - just different token counts per sequence.

## Attention Ranges: Which K,V Does Each Q Group Attend To?

This is the **critical** part that makes the fused kernel work correctly with causal masking.

### Chunk-Based Attention

The full K,V sequence is divided into **chunks**:

```python
# Full K,V sequence (after all-gather across world_size ranks)
total_chunks = 2 * world_size

# Example: world_size = 4 → total_chunks = 8
# K,V is divided into 8 equal chunks:
#
#   K,V: [chunk0][chunk1][chunk2][chunk3][chunk4][chunk5][chunk6][chunk7]
#        <─────────────────── full sequence ────────────────────────────>
```

### Q0 and Q1 Represent Different Chunks

- **Q0** represents tokens at chunk position `chunk_idx_0` (early)
- **Q1** represents tokens at chunk position `chunk_idx_1` (late)

```python
# Example: Rank 0 in world_size=4
chunk_idx_0 = 0   # Q0 is at chunk position 0 (earliest)
chunk_idx_1 = 7   # Q1 is at chunk position 7 (latest)
total_chunks = 8
```

### Causal Masking Determines Attention Range

With causal attention, each Q token can only attend to K,V tokens **up to and including its chunk position**.

```python
# Chunk size (tokens per chunk)
chunk_size = total_k_tokens // total_chunks

# Q0 attention range (up to chunk_idx_0)
end_n0 = (chunk_idx_0 + 1) * chunk_size
q0_attends_to = k[0:end_n0], v[0:end_n0]

# Q1 attention range (up to chunk_idx_1)
end_n1 = (chunk_idx_1 + 1) * chunk_size
q1_attends_to = k[0:end_n1], v[0:end_n1]
```

### Visual Example: Rank 0, world_size=4

```
Full K,V sequence (8 chunks):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ chunk0 │ chunk1 │ chunk2 │ chunk3 │ chunk4 │ chunk5 │ chunk6 │ chunk7 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
  ^                                                               ^
  │                                                               │
  chunk_idx_0 = 0                                      chunk_idx_1 = 7

Q0 attends to:
┌────────┐
│ chunk0 │  ← Only first chunk (causal: up to position 0)
└────────┘

Q1 attends to:
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ chunk0 │ chunk1 │ chunk2 │ chunk3 │ chunk4 │ chunk5 │ chunk6 │ chunk7 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
 ← All chunks (causal: up to position 7, which is the last chunk)
```

### Visual Example: Rank 2, world_size=4

```
Full K,V sequence (8 chunks):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ chunk0 │ chunk1 │ chunk2 │ chunk3 │ chunk4 │ chunk5 │ chunk6 │ chunk7 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
                    ^                           ^
                    │                           │
          chunk_idx_0 = 2               chunk_idx_1 = 5

Q0 attends to:
┌────────┬────────┬────────┐
│ chunk0 │ chunk1 │ chunk2 │  ← First 3 chunks (causal: up to position 2)
└────────┴────────┴────────┘

Q1 attends to:
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ chunk0 │ chunk1 │ chunk2 │ chunk3 │ chunk4 │ chunk5 │  ← First 6 chunks
└────────┴────────┴────────┴────────┴────────┴────────┘
 ← Causal: up to position 5
```

### Attention Pattern Per Sequence

For each sequence in the batch, the same chunk-based pattern applies:

```python
# For sequence i in batch:
seq_start_k = cu_seqlens_k[i]
seq_end_k = cu_seqlens_k[i + 1]
seq_len_k = seq_end_k - seq_start_k

# Chunk size for this sequence
chunk_size_i = seq_len_k // total_chunks

# Q0 attention range for sequence i
end_n0_i = (chunk_idx_0 + 1) * chunk_size_i
q0_k_range = k[seq_start_k : seq_start_k + end_n0_i]
q0_v_range = v[seq_start_k : seq_start_k + end_n0_i]

# Q1 attention range for sequence i
end_n1_i = (chunk_idx_1 + 1) * chunk_size_i
q1_k_range = k[seq_start_k : seq_start_k + end_n1_i]
q1_v_range = v[seq_start_k : seq_start_k + end_n1_i]
```

### Concrete Example with Numbers

```python
# Configuration
world_size = 4
total_chunks = 8
rank = 0
chunk_idx_0 = 0   # Q0 at position 0
chunk_idx_1 = 7   # Q1 at position 7

# Batch with 1 sequence
batch_size = 1

# After all-gather, sequence has 800 tokens
cu_seqlens_k = torch.tensor([0, 800], dtype=torch.int32)
seq_len_k = 800

# Chunk size
chunk_size = 800 // 8 = 100 tokens per chunk

# Q0 attention range
end_n0 = (0 + 1) * 100 = 100
# Q0 attends to: k[0:100], v[0:100]
# This is chunk0 only

# Q1 attention range
end_n1 = (7 + 1) * 100 = 800
# Q1 attends to: k[0:800], v[0:800]
# This is all 8 chunks (chunk0 through chunk7)

# Q0 has 100 tokens (1 chunk worth)
cu_seqlens_q0 = torch.tensor([0, 100], dtype=torch.int32)

# Q1 has 100 tokens (1 chunk worth)
cu_seqlens_q1 = torch.tensor([0, 100], dtype=torch.int32)
```

### Why This Matters for the Fused Kernel

The kernel needs to:

1. **Load K,V blocks once** and reuse them for both Q0 and Q1
2. **Apply different causal masks**:
   - For Q0: mask out K,V beyond `end_n0`
   - For Q1: mask out K,V beyond `end_n1`
3. **Skip unnecessary K,V blocks**:
   - If processing Q0 and current K,V block is beyond `end_n0`, skip Q0 computation
   - If processing Q1 and current K,V block is beyond `end_n1`, skip Q1 computation

```python
# Kernel pseudocode
for k_block_start in range(0, max(end_n0, end_n1), BLOCK_N):
    # Load K,V block ONCE
    k_block = load(k[k_block_start:k_block_start + BLOCK_N])
    v_block = load(v[k_block_start:k_block_start + BLOCK_N])

    # Use for Q0 if within range
    if k_block_start < end_n0:
        compute_attention(q0, k_block, v_block, causal_mask_for_q0)

    # Use for Q1 if within range (REUSING same k_block, v_block!)
    if k_block_start < end_n1:
        compute_attention(q1, k_block, v_block, causal_mask_for_q1)
```

### Summary: Attention Ranges

| Group | Chunk Position | Attention Range | Example (8 chunks) |
|-------|----------------|-----------------|-------------------|
| Q0 | `chunk_idx_0` | K,V[0 : (chunk_idx_0+1) * chunk_size] | chunk_idx_0=0 → chunks [0] |
| Q1 | `chunk_idx_1` | K,V[0 : (chunk_idx_1+1) * chunk_size] | chunk_idx_1=7 → chunks [0-7] |

**Key Insight**:
- Q0 (early chunks) has **restricted** attention to early K,V
- Q1 (late chunks) has **broader** attention to most/all K,V
- Both share the **same K,V buffer**, just with different valid ranges
- The fused kernel loads K,V **once** and applies both range restrictions

## API Design

### Function Signature

```python
def fused_zigzag_llama3_flash_attn_varlen_forward(
    # Q groups (two different Q sequences)
    q0: torch.Tensor,              # [total_q0_tokens, nheads, headdim]
    q1: torch.Tensor,              # [total_q1_tokens, nheads, headdim]

    # Shared K,V
    k: torch.Tensor,               # [total_k_tokens, nheads_k, headdim]
    v: torch.Tensor,               # [total_v_tokens, nheads_k, headdim]

    # Cumulative sequence lengths (varlen)
    cu_seqlens_q0: torch.Tensor,   # [batch_size + 1] - boundaries for Q0
    cu_seqlens_q1: torch.Tensor,   # [batch_size + 1] - boundaries for Q1
    cu_seqlens_k: torch.Tensor,    # [batch_size + 1] - boundaries for K,V

    # Metadata
    max_seqlen_q0: int,            # max(cu_seqlens_q0[i+1] - cu_seqlens_q0[i])
    max_seqlen_q1: int,            # max(cu_seqlens_q1[i+1] - cu_seqlens_q1[i])
    max_seqlen_k: int,             # max(cu_seqlens_k[i+1] - cu_seqlens_k[i])

    # Attention range parameters (kernel is agnostic to distributed details)
    max_kv_len_q0: int,            # Max K,V length Q0 can attend to (per sequence)
    max_kv_len_q1: int,            # Max K,V length Q1 can attend to (per sequence)

    # Standard attention parameters
    softmax_scale: Optional[float] = None,
    causal: bool = True,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        out0: [total_q0_tokens, nheads, headdim] - output for Q0
        out1: [total_q1_tokens, nheads, headdim] - output for Q1
        lse0: [nheads, total_q0_tokens] - log-sum-exp for Q0
        lse1: [nheads, total_q1_tokens] - log-sum-exp for Q1
    """
```

### Input Shapes and Relationships

For a batch of `B` sequences:

```python
batch_size = B

# Q0 sequences (early chunks)
cu_seqlens_q0 = [0, len0, len0+len1, len0+len1+len2, ...]  # B+1 elements
total_q0_tokens = cu_seqlens_q0[B]  # Last element
q0.shape = [total_q0_tokens, nheads, headdim]

# Q1 sequences (late chunks)
cu_seqlens_q1 = [0, len0', len0'+len1', len0'+len1'+len2', ...]  # B+1 elements
total_q1_tokens = cu_seqlens_q1[B]
q1.shape = [total_q1_tokens, nheads, headdim]

# K,V sequences (full sequences after all-gather)
cu_seqlens_k = [0, klen0, klen0+klen1, klen0+klen1+klen2, ...]  # B+1 elements
total_k_tokens = cu_seqlens_k[B]
k.shape = [total_k_tokens, nheads_k, headdim]
v.shape = [total_k_tokens, nheads_k, headdim]

# Important: All cu_seqlens arrays have B+1 elements
assert len(cu_seqlens_q0) == batch_size + 1
assert len(cu_seqlens_q1) == batch_size + 1
assert len(cu_seqlens_k) == batch_size + 1
```

### Example with Real Numbers

```python
# Batch of 2 sequences
batch_size = 2

# Sequence 0: 100 tokens total, split as 30 (Q0) + 70 (Q1)
# Sequence 1: 200 tokens total, split as 80 (Q0) + 120 (Q1)

# After all-gather (world_size=4):
# Sequence 0: 400 tokens in K,V
# Sequence 1: 800 tokens in K,V

# Q0 (early chunks): concatenated [seq0_early, seq1_early]
cu_seqlens_q0 = torch.tensor([0, 30, 110], dtype=torch.int32)
# seq0: q0[0:30]
# seq1: q0[30:110]
q0.shape = [110, 32, 128]  # [total_tokens, nheads, headdim]

# Q1 (late chunks): concatenated [seq0_late, seq1_late]
cu_seqlens_q1 = torch.tensor([0, 70, 190], dtype=torch.int32)
# seq0: q1[0:70]
# seq1: q1[70:190]
q1.shape = [190, 32, 128]

# K,V (full sequences): concatenated [seq0_full, seq1_full]
cu_seqlens_k = torch.tensor([0, 400, 1200], dtype=torch.int32)
# seq0: k[0:400], v[0:400]
# seq1: k[400:1200], v[400:1200]
k.shape = [1200, 8, 128]  # [total_tokens, nheads_k, headdim] (GQA: 8 KV heads)
v.shape = [1200, 8, 128]

# Compute attention ranges (caller handles distributed logic)
chunk_size = 1200 // 8  # 150 tokens per chunk
max_kv_len_q0 = (0 + 1) * chunk_size  # 150 (chunk 0)
max_kv_len_q1 = (7 + 1) * chunk_size  # 1200 (all chunks)

# Call fused kernel
out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
    q0, q1, k, v,
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_seqlen_q0=80,   # max(30, 80)
    max_seqlen_q1=120,  # max(70, 120)
    max_seqlen_k=800,   # max(400, 800)
    max_kv_len_q0=150,  # Q0 attends to first 150 tokens
    max_kv_len_q1=1200, # Q1 attends to all 1200 tokens
    causal=True,
)

# Outputs have same shape as inputs
out0.shape = [110, 32, 128]  # same as q0
out1.shape = [190, 32, 128]  # same as q1
```

## Kernel Execution Logic

For each sequence `i` in the batch (0 to B-1):

```python
# 1. Extract sequence boundaries
q0_start = cu_seqlens_q0[i]
q0_end = cu_seqlens_q0[i + 1]
q0_len = q0_end - q0_start

q1_start = cu_seqlens_q1[i]
q1_end = cu_seqlens_q1[i + 1]
q1_len = q1_end - q1_start

k_start = cu_seqlens_k[i]
k_end = cu_seqlens_k[i + 1]
k_len = k_end - k_start

# 2. Determine attention ranges
# These are passed in as parameters (kernel doesn't compute from chunks)
end_n0 = min(max_kv_len_q0, k_len)  # Q0's K,V range end
end_n1 = min(max_kv_len_q1, k_len)  # Q1's K,V range end
# Q0 attends to: k_seq[0:end_n0], v_seq[0:end_n0]
# Q1 attends to: k_seq[0:end_n1], v_seq[0:end_n1]

# Maximum range we need to iterate over
max_end_n = max(end_n0, end_n1)

# 3. Extract this sequence's data
q0_seq = q0[q0_start:q0_end]  # [q0_len, nheads, headdim]
q1_seq = q1[q1_start:q1_end]  # [q1_len, nheads, headdim]
k_seq = k[k_start:k_end]      # [k_len, nheads_k, headdim]
v_seq = v[k_start:k_end]      # [k_len, nheads_k, headdim]

# 4. Fused computation - iterate over K,V blocks ONCE
for n_block_start in range(0, max_end_n, BLOCK_N):
    # Load K,V block ONCE (shared between Q0 and Q1)
    k_block = load_block(k_seq, n_block_start, BLOCK_N)  # [BLOCK_N, headdim]
    v_block = load_block(v_seq, n_block_start, BLOCK_N)  # [BLOCK_N, headdim]

    # Process Q0 with this K,V block (if within Q0's attention range)
    if n_block_start < end_n0:
        for m_block in range(0, q0_len, BLOCK_M):
            q0_block = q0_seq[m_block:m_block+BLOCK_M]

            # Compute Q0 @ K^T
            qk = matmul(q0_block, k_block.T) * softmax_scale

            # Apply causal mask (Q can't attend to future positions)
            # Also apply range mask (Q0 can only attend up to end_n0)
            mask = create_mask(m_block, n_block_start, end_n0, causal)
            qk_masked = where(mask, qk, -inf)

            # Softmax and accumulate
            attn = softmax(qk_masked)
            acc_o0 += matmul(attn, v_block)

    # Process Q1 with SAME K,V block (if within Q1's attention range)
    if n_block_start < end_n1:
        for m_block in range(0, q1_len, BLOCK_M):
            q1_block = q1_seq[m_block:m_block+BLOCK_M]

            # Compute Q1 @ K^T (REUSING same k_block!)
            qk = matmul(q1_block, k_block.T) * softmax_scale

            # Apply causal mask (Q can't attend to future positions)
            # Also apply range mask (Q1 can only attend up to end_n1)
            mask = create_mask(m_block, n_block_start, end_n1, causal)
            qk_masked = where(mask, qk, -inf)

            # Softmax and accumulate
            attn = softmax(qk_masked)
            acc_o1 += matmul(attn, v_block)  # REUSING same v_block!

# 5. Write outputs
out0[q0_start:q0_end] = result0
out1[q1_start:q1_end] = result1
```

### Key Points in Execution

1. **Single K,V loop**: We iterate over K,V blocks once (up to `max_end_n`)
2. **Conditional processing**:
   - Q0 computation only runs if `n_block_start < end_n0`
   - Q1 computation only runs if `n_block_start < end_n1`
3. **Shared K,V blocks**: The same `k_block` and `v_block` are used for both Q0 and Q1
4. **Different causal masks**: Q0 and Q1 have different chunk positions, so causal masks differ
5. **Memory savings**: K,V read from HBM **once** instead of twice

## Grid and Thread Configuration

```python
# Grid dimensions
batch_size = len(cu_seqlens_k) - 1
grid = (
    triton.cdiv(max(max_seqlen_q0, max_seqlen_q1), BLOCK_M),  # M dimension
    batch_size * nheads,                                       # Batch * Heads
)

# Each kernel instance processes:
# - One M-block (BLOCK_M rows of Q)
# - One sequence in batch
# - One head
# - Both Q groups (Q0 and Q1) for that M-block
```

## Visual: Fused Kernel Memory Access Pattern

This diagram shows how the fused kernel processes K,V blocks efficiently:

```
Time →

K,V Blocks:  [Block0][Block1][Block2][Block3][Block4][Block5][Block6][Block7]
             ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓
Load from    Load    Load    Load    Load    Load    Load    Load    Load
HBM once:    once    once    once    once    once    once    once    once

Use for Q0:  ✓       ✗       ✗       ✗       ✗       ✗       ✗       ✗
(chunk_idx_0=0)      (skip - beyond end_n0)
             ↓
Use for Q1:  ✓       ✓       ✓       ✓       ✓       ✓       ✓       ✓
(chunk_idx_1=7)      (all blocks within end_n1)
             ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓

Result:      K,V blocks loaded ONCE from HBM, used for BOTH Q0 and Q1
             50% memory bandwidth savings vs. loading separately!
```

Compare to original (two separate calls):

```
Call 1 (Q0):
K,V Blocks:  [Block0][Block1][Block2][Block3][Block4][Block5][Block6][Block7]
             ↓
Load from    Load    (skip remaining blocks)
HBM:         once
Use for Q0:  ✓

Call 2 (Q1):
K,V Blocks:  [Block0][Block1][Block2][Block3][Block4][Block5][Block6][Block7]
             ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓
Load from    Load    Load    Load    Load    Load    Load    Load    Load
HBM AGAIN:   AGAIN!  AGAIN!  AGAIN!  AGAIN!  AGAIN!  AGAIN!  AGAIN!  AGAIN!
Use for Q1:  ✓       ✓       ✓       ✓       ✓       ✓       ✓       ✓

Result:      Block0 loaded TWICE - wasteful!
             Total: 1 + 8 = 9 block loads vs. 8 in fused version
```

## Comparison: Original vs Fused

### Original (Two Calls)

```python
# Call 1: Q0 group
out0, lse0 = _flash_attn_varlen_forward(
    q0, k, v,
    cu_seqlens_q0, cu_seqlens_k,
    max_seqlen_q0, max_seqlen_k,
    ...
)
# Memory reads: Q0 (1x), K (1x), V (1x)

# Call 2: Q1 group
out1, lse1 = _flash_attn_varlen_forward(
    q1, k, v,
    cu_seqlens_q1, cu_seqlens_k,
    max_seqlen_q1, max_seqlen_k,
    ...
)
# Memory reads: Q1 (1x), K (1x), V (1x)

# Total memory reads:
# Q0: 1x
# Q1: 1x
# K: 2x ← REDUNDANT!
# V: 2x ← REDUNDANT!
```

### Fused (Single Call)

```python
# Single fused call
out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
    q0, q1, k, v,
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
    chunk_idx_0, chunk_idx_1, total_chunks,
    ...
)

# Total memory reads:
# Q0: 1x
# Q1: 1x
# K: 1x ← SAVED 50%!
# V: 1x ← SAVED 50%!
```

## Implementation Requirements

### Kernel Requirements

1. **Support varlen format**: Handle cu_seqlens for variable-length sequences
2. **Process both Q groups**: Compute attention for Q0 and Q1 in same kernel
3. **Load K,V once**: Share loaded K,V blocks between Q0 and Q1 computations
4. **Independent causal masking**: Different K,V ranges for Q0 vs Q1 based on chunk indices
5. **Independent LSE tracking**: Separate log-sum-exp statistics for each Q group

### Memory Layout

```
Q0: [total_q0_tokens, nheads, headdim]
    ├─ Sequence 0: [cu_seqlens_q0[0]:cu_seqlens_q0[1]]
    ├─ Sequence 1: [cu_seqlens_q0[1]:cu_seqlens_q0[2]]
    └─ ...

Q1: [total_q1_tokens, nheads, headdim]
    ├─ Sequence 0: [cu_seqlens_q1[0]:cu_seqlens_q1[1]]
    ├─ Sequence 1: [cu_seqlens_q1[1]:cu_seqlens_q1[2]]
    └─ ...

K,V: [total_k_tokens, nheads_k, headdim]
     ├─ Sequence 0: [cu_seqlens_k[0]:cu_seqlens_k[1]]
     ├─ Sequence 1: [cu_seqlens_k[1]:cu_seqlens_k[2]]
     └─ ...
```

## Key Differences from Standard Flash Attention

| Aspect | Standard Flash Attn | Fused Zigzag Attn |
|--------|-------------------|-------------------|
| **Q inputs** | 1 tensor | 2 tensors (Q0, Q1) |
| **K,V inputs** | 1 pair | 1 pair (shared) |
| **Outputs** | 1 output, 1 LSE | 2 outputs, 2 LSEs |
| **cu_seqlens** | 2 arrays (q, k) | 3 arrays (q0, q1, k) |
| **Batch semantics** | Independent sequences | Same sequences, split differently |
| **Kernel calls** | 1 | 1 (vs 2 in naive approach) |
| **K,V memory reads** | 1x | 1x (vs 2x in naive) |

## Summary

**Core Idea**: Merge two flash attention calls that use the same K,V but different Q sequences.

**API Design Principles**:
1. Three cu_seqlens arrays for Q0, Q1, and K (all with same batch size)
2. All inputs are in varlen format (concatenated sequences)
3. Same batch size across all inputs (same sequences, just split differently)
4. Two separate outputs and LSE tensors (one per Q group)
5. Single kernel launch processes both Q groups

**Performance Win**: 50% reduction in K,V memory bandwidth (2x → 1x reads)

**Compatibility**: Drop-in replacement for original `execute_grouped_attention` function.
