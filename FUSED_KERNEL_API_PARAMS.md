# Fused Kernel API Parameters - Quick Reference

## Function Signature

```python
def fused_zigzag_llama3_flash_attn_varlen_forward(
    q0, q1, k, v,
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
    max_kv_len_q0, max_kv_len_q1,
    softmax_scale=None,
    causal=True,
) -> (out0, out1, lse0, lse1)
```

## Input Tensors

### `q0: torch.Tensor`
- **Shape**: `[total_q0_tokens, nheads, headdim]`
- **What**: Query tokens for **early chunks** (group 0)
- **Format**: Varlen - concatenated sequences from all batch items
- **Example**: If batch has 2 sequences with Q0 having 30 and 80 tokens respectively → shape `[110, 32, 128]`

### `q1: torch.Tensor`
- **Shape**: `[total_q1_tokens, nheads, headdim]`
- **What**: Query tokens for **late chunks** (group 1)
- **Format**: Varlen - concatenated sequences from all batch items
- **Example**: If batch has 2 sequences with Q1 having 70 and 120 tokens respectively → shape `[190, 32, 128]`
- **Note**: Same sequences as Q0, just different tokens (complementary split)

### `k: torch.Tensor`
- **Shape**: `[total_k_tokens, nheads_k, headdim]`
- **What**: Key tokens (after all-gather, full sequences)
- **Format**: Varlen - concatenated sequences
- **Shared by**: Both Q0 and Q1 use the same K
- **Example**: 2 sequences with 400 and 800 tokens → shape `[1200, 8, 128]` (GQA: 8 K heads)

### `v: torch.Tensor`
- **Shape**: `[total_v_tokens, nheads_k, headdim]`
- **What**: Value tokens (after all-gather, full sequences)
- **Format**: Same as K
- **Shared by**: Both Q0 and Q1 use the same V
- **Note**: `total_v_tokens == total_k_tokens` always

## Sequence Boundaries (Varlen Format)

### `cu_seqlens_q0: torch.Tensor`
- **Shape**: `[batch_size + 1]`
- **Dtype**: `torch.int32`
- **What**: **Cu**mulative **seq**uence **len**gth**s** for Q0
- **Purpose**: Tells where each sequence starts/ends in the Q0 tensor
- **Format**: `[0, len_seq0, len_seq0+len_seq1, len_seq0+len_seq1+len_seq2, ...]`
- **Example**: `[0, 30, 110]` means:
  - Sequence 0: `q0[0:30]` (30 tokens)
  - Sequence 1: `q0[30:110]` (80 tokens)
- **Usage**: `seq_start = cu_seqlens_q0[i]`, `seq_end = cu_seqlens_q0[i+1]`

### `cu_seqlens_q1: torch.Tensor`
- **Shape**: `[batch_size + 1]`
- **Dtype**: `torch.int32`
- **What**: Cumulative sequence lengths for Q1
- **Same sequences as Q0**: Just different token counts per sequence
- **Example**: `[0, 70, 190]` means:
  - Sequence 0: `q1[0:70]` (70 tokens)
  - Sequence 1: `q1[70:190]` (120 tokens)

### `cu_seqlens_k: torch.Tensor`
- **Shape**: `[batch_size + 1]`
- **Dtype**: `torch.int32`
- **What**: Cumulative sequence lengths for K,V
- **Same for both**: K and V have identical sequence boundaries
- **Example**: `[0, 400, 1200]` means:
  - Sequence 0: `k[0:400]`, `v[0:400]` (400 tokens)
  - Sequence 1: `k[400:1200]`, `v[400:1200]` (800 tokens)

**Critical**: All three `cu_seqlens_*` arrays have the **same batch_size** (same number of sequences, just different token counts)

## Max Sequence Lengths

### `max_seqlen_q0: int`
- **What**: Maximum sequence length in Q0 across the batch
- **Formula**: `max(cu_seqlens_q0[i+1] - cu_seqlens_q0[i])` for all i
- **Purpose**: Kernel needs this to allocate proper grid size
- **Example**: If Q0 sequences are [30, 80] → `max_seqlen_q0 = 80`

### `max_seqlen_q1: int`
- **What**: Maximum sequence length in Q1 across the batch
- **Example**: If Q1 sequences are [70, 120] → `max_seqlen_q1 = 120`

### `max_seqlen_k: int`
- **What**: Maximum sequence length in K,V across the batch
- **Example**: If K sequences are [400, 800] → `max_seqlen_k = 800`

**Why needed**: Triton kernel grid size is `(cdiv(max_seqlen, BLOCK_M), batch_size * nheads)`

## Attention Ranges (Simplified - No Distributed Details)

### `max_kv_len_q0: int`
- **What**: Maximum K,V length that Q0 can attend to (per sequence)
- **Range**: `0` to `max_seqlen_k`
- **Purpose**: Limits how far into K,V that Q0 tokens can attend (causal masking)
- **Example**: If Q0 should only attend to first 100 tokens → `max_kv_len_q0 = 100`
- **For each sequence**: Q0 attends to `k[seq_start : seq_start + max_kv_len_q0]`

### `max_kv_len_q1: int`
- **What**: Maximum K,V length that Q1 can attend to (per sequence)
- **Range**: `0` to `max_seqlen_k`
- **Purpose**: Limits how far into K,V that Q1 tokens can attend (causal masking)
- **Example**: If Q1 should attend to all 800 tokens → `max_kv_len_q1 = 800`
- **For each sequence**: Q1 attends to `k[seq_start : seq_start + max_kv_len_q1]`

**How caller computes these (from chunk info, if using zigzag)**:
```python
# Caller computes from distributed training info (kernel doesn't need to know):
chunk_size = max_seqlen_k // total_chunks
max_kv_len_q0 = (chunk_idx_0 + 1) * chunk_size  # e.g., 100 for chunk 0
max_kv_len_q1 = (chunk_idx_1 + 1) * chunk_size  # e.g., 800 for chunk 7

# Then pass to kernel (kernel only sees the ranges, not chunks)
fused_kernel(..., max_kv_len_q0, max_kv_len_q1, ...)
```

**Key point**: The kernel is **agnostic to distributed training** - it just knows "Q0 attends up to position X, Q1 attends up to position Y"

## Standard Attention Parameters

### `softmax_scale: Optional[float]`
- **What**: Scaling factor for Q @ K^T before softmax
- **Default**: `1.0 / sqrt(headdim)`
- **Purpose**: Prevents saturation in softmax for large dot products
- **Example**: For `headdim=128` → default is `1/sqrt(128) ≈ 0.0884`

### `causal: bool`
- **What**: Whether to apply causal masking
- **Default**: `True`
- **Meaning**:
  - `True`: Each Q token can only attend to K,V tokens at same or earlier positions
  - `False`: Full bidirectional attention
- **For zigzag**: Always `True` in practice

## Return Values

### `out0: torch.Tensor`
- **Shape**: `[total_q0_tokens, nheads, headdim]` (same as q0)
- **What**: Attention output for Q0
- **Format**: Varlen - concatenated outputs for all sequences
- **Use**: Scatter back to original positions using `chunk_indices_list[0]`

### `out1: torch.Tensor`
- **Shape**: `[total_q1_tokens, nheads, headdim]` (same as q1)
- **What**: Attention output for Q1
- **Use**: Scatter back to original positions using `chunk_indices_list[1]`

### `lse0: torch.Tensor`
- **Shape**: `[nheads, total_q0_tokens]` or `[batch_size, nheads, max_seqlen_q0_rounded]`
- **What**: **L**og-**S**um-**E**xp statistics for Q0
- **Purpose**: Used in backward pass for gradient computation
- **Contains**: `log(sum(exp(attention_scores)))` for each query position

### `lse1: torch.Tensor`
- **Shape**: `[nheads, total_q1_tokens]` or `[batch_size, nheads, max_seqlen_q1_rounded]`
- **What**: Log-Sum-Exp statistics for Q1
- **Purpose**: Same as lse0, but for Q1

## Quick Example: All Parameters

```python
# Setup
batch_size = 2  # 2 sequences in batch

# Inputs
q0 = torch.randn([110, 32, 128])  # [total_q0_tokens=30+80, nheads=32, headdim=128]
q1 = torch.randn([190, 32, 128])  # [total_q1_tokens=70+120, nheads=32, headdim=128]
k = torch.randn([1200, 8, 128])   # [total_k_tokens=400+800, nheads_k=8, headdim=128]
v = torch.randn([1200, 8, 128])   # [total_v_tokens=400+800, nheads_k=8, headdim=128]

# Sequence boundaries
cu_seqlens_q0 = torch.tensor([0, 30, 110], dtype=torch.int32)      # batch_size + 1 = 3
cu_seqlens_q1 = torch.tensor([0, 70, 190], dtype=torch.int32)      # batch_size + 1 = 3
cu_seqlens_k = torch.tensor([0, 400, 1200], dtype=torch.int32)     # batch_size + 1 = 3

# Max lengths
max_seqlen_q0 = 80   # max(30, 80)
max_seqlen_q1 = 120  # max(70, 120)
max_seqlen_k = 800   # max(400, 800)

# Attention ranges
# Q0 should attend to first 100 tokens of K,V (e.g., from chunk computation)
max_kv_len_q0 = 100

# Q1 should attend to all 800 tokens of K,V
max_kv_len_q1 = 800

# Standard params
softmax_scale = 1.0 / math.sqrt(128)  # ≈ 0.0884
causal = True

# Call
out0, out1, lse0, lse1 = fused_zigzag_llama3_flash_attn_varlen_forward(
    q0, q1, k, v,
    cu_seqlens_q0, cu_seqlens_q1, cu_seqlens_k,
    max_seqlen_q0, max_seqlen_q1, max_seqlen_k,
    max_kv_len_q0, max_kv_len_q1,
    softmax_scale,
    causal,
)

# Outputs
out0.shape  # [110, 32, 128] - same as q0
out1.shape  # [190, 32, 128] - same as q1
lse0.shape  # [32, 110] or [2, 32, 128] (rounded)
lse1.shape  # [32, 190] or [2, 32, 128] (rounded)
```

### How Caller Computes Attention Ranges (Example for Zigzag)

```python
# If using zigzag ring attention, caller computes ranges from chunk info:
world_size = 4
rank = 0
total_chunks = 2 * world_size  # = 8

chunk_idx_0 = rank                        # = 0
chunk_idx_1 = 2 * world_size - 1 - rank  # = 7

chunk_size = max_seqlen_k // total_chunks  # 800 // 8 = 100

# Compute attention ranges
max_kv_len_q0 = (chunk_idx_0 + 1) * chunk_size  # (0 + 1) * 100 = 100
max_kv_len_q1 = (chunk_idx_1 + 1) * chunk_size  # (7 + 1) * 100 = 800

# Kernel doesn't need to know about world_size, chunks, or rank!
```

## Key Relationships

```python
# All cu_seqlens have same batch size
len(cu_seqlens_q0) == len(cu_seqlens_q1) == len(cu_seqlens_k) == batch_size + 1

# Total tokens match tensor shapes
cu_seqlens_q0[-1] == total_q0_tokens == q0.shape[0]
cu_seqlens_q1[-1] == total_q1_tokens == q1.shape[0]
cu_seqlens_k[-1] == total_k_tokens == k.shape[0] == v.shape[0]

# Max lengths
max_seqlen_q0 >= all individual Q0 sequence lengths
max_seqlen_q1 >= all individual Q1 sequence lengths
max_seqlen_k >= all individual K,V sequence lengths

# Attention ranges
0 <= max_kv_len_q0 <= max_seqlen_k
0 <= max_kv_len_q1 <= max_seqlen_k
max_kv_len_q0 <= max_kv_len_q1  (usually Q0 is early, Q1 is late)

# Per sequence, attention ranges are:
# Q0 attends to: k[seq_start : seq_start + max_kv_len_q0]
# Q1 attends to: k[seq_start : seq_start + max_kv_len_q1]
```

## Summary Table

| Parameter | Type | Shape/Range | Purpose |
|-----------|------|-------------|---------|
| **q0** | Tensor | `[total_q0_tokens, nheads, headdim]` | Query group 0 |
| **q1** | Tensor | `[total_q1_tokens, nheads, headdim]` | Query group 1 |
| **k** | Tensor | `[total_k_tokens, nheads_k, headdim]` | Keys (shared by Q0 and Q1) |
| **v** | Tensor | `[total_v_tokens, nheads_k, headdim]` | Values (shared by Q0 and Q1) |
| **cu_seqlens_q0** | Tensor | `[batch_size + 1]` int32 | Q0 sequence boundaries |
| **cu_seqlens_q1** | Tensor | `[batch_size + 1]` int32 | Q1 sequence boundaries |
| **cu_seqlens_k** | Tensor | `[batch_size + 1]` int32 | K,V sequence boundaries |
| **max_seqlen_q0** | int | `> 0` | Max Q0 sequence length |
| **max_seqlen_q1** | int | `> 0` | Max Q1 sequence length |
| **max_seqlen_k** | int | `> 0` | Max K,V sequence length |
| **max_kv_len_q0** | int | `0 to max_seqlen_k` | Max K,V range for Q0 |
| **max_kv_len_q1** | int | `0 to max_seqlen_k` | Max K,V range for Q1 |
| **softmax_scale** | float | Optional, default `1/√headdim` | Softmax scaling |
| **causal** | bool | True/False | Causal masking enable |
| **out0** | Tensor | `[total_q0_tokens, nheads, headdim]` | Q0 attention output |
| **out1** | Tensor | `[total_q1_tokens, nheads, headdim]` | Q1 attention output |
| **lse0** | Tensor | `[nheads, total_q0_tokens]` | Q0 logsumexp stats |
| **lse1** | Tensor | `[nheads, total_q1_tokens]` | Q1 logsumexp stats |
