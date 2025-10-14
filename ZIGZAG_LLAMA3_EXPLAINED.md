# Zigzag Llama3 Flash Attention - Complete Guide

A comprehensive guide to understanding the zigzag_llama3 flash attention algorithm, which combines zigzag distribution for load balancing with Llama3-style single-step all-gather communication.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Data Distribution](#data-distribution)
4. [Forward Pass](#forward-pass)
5. [Backward Pass](#backward-pass)
6. [Execution Modes](#execution-modes)
7. [Load Balancing](#load-balancing)
8. [Complete Example](#complete-example)

---

## Overview

### What is Zigzag Llama3?

Zigzag Llama3 is a distributed flash attention algorithm designed to address load imbalance in causal attention:
- Uses **zigzag interleaved distribution** to theoretically balance workload across GPUs
- Uses **Llama3-style all-gather** (single communication step) for K,V
- Splits queries by **global chunk index** (not position) for efficient causal attention
- Supports both **two-kernels** (sequential) and **fused** (single kernel) execution modes

### Motivation: The Load Imbalance Problem

**Problem with Standard Contiguous Distribution:**
In causal attention, early tokens attend to few tokens (light work), while late tokens attend to many tokens (heavy work). With contiguous distribution, some GPUs get all light work, others get all heavy work, leading to:
- GPU idle time (fast GPUs wait for slow GPUs)
- Poor overall throughput (bottlenecked by slowest GPU)
- Wasted compute resources

**Proposed Zigzag Solution:**
Each GPU gets tokens from both the beginning (light work) and end (heavy work) of sequences. Theoretically, this should:
- Balance computational load across all GPUs
- Eliminate GPU idle time
- Improve overall throughput

**This document explains the algorithm and analyzes the theoretical benefits to be validated through benchmarking.**

---

## Key Concepts

### 1. Zigzag Interleaved Distribution

**Pattern:** Each GPU gets 2 chunks: one from the beginning, one from the end.

```
World Size = 4
Total Chunks = 8 (2 × world_size)

GPU 0: chunk[0] + chunk[7]  (first + last)
GPU 1: chunk[1] + chunk[6]  (second + second-to-last)
GPU 2: chunk[2] + chunk[5]  (third + third-to-last)
GPU 3: chunk[3] + chunk[4]  (fourth + fourth-to-last)
```

### 2. Chunk Index Groups

Chunks are categorized into **early** and **late** groups:

```
Early group: chunks [0, 1, 2, 3]      (indices < world_size)
Late group:  chunks [4, 5, 6, 7]      (indices >= world_size)
```

**Causality Rule:**

To maintain causality, a query from a chunk with global index `i` attends to K,V from all chunks up to and including chunk `i`.

For example:
- Queries from `chunk[2]` attend to K,V from `[c0, c1, c2]`
- Queries from `chunk[5]` attend to K,V from `[c0, c1, c2, c3, c4, c5]`

The implementation groups queries by "early" and "late" chunks for efficiency, but the core causality is respected on a per-chunk basis. This creates the load balancing benefit!

### 3. Llama3-Style All-Gather

**Single-step communication:**
```
Before:  Each GPU has local K,V (2 chunks)
After:   Each GPU has ALL K,V (8 chunks)
```

Unlike ring attention (N steps), this completes in 1 step, hiding communication latency with computation.

---

## Data Distribution

### Input Format

Sequences arrive in **zigzag interleaved format**:

```python
# Example: 1 sequence, 8 chunks, world_size=4

Full sequence:  [c0, c1, c2, c3, c4, c5, c6, c7]

# Distribution to GPUs:
GPU 0: [c0, c7]  (concatenated together)
GPU 1: [c1, c6]
GPU 2: [c2, c5]
GPU 3: [c3, c4]
```

### How to Create Zigzag Distribution

```python
def extract_local(value, cu_seqlens, rank, world_size):
    """Extract local zigzag-distributed portion for this rank."""
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()

        # Split sequence into 2*world_size chunks
        chunks = value[start:end].chunk(2 * world_size, dim=0)

        # Take chunk[rank] and chunk[2*world_size - 1 - rank]
        local_values.extend([
            chunks[rank],
            chunks[2 * world_size - 1 - rank],
        ])

    return torch.cat(local_values, dim=0)

# Usage:
local_q = extract_local(q, cu_seqlens, rank, world_size)
local_kv = extract_local(kv, cu_seqlens, rank, world_size)
```

---

## Forward Pass

### Overview

The forward pass transforms local Q, K, V into attention outputs through 7 steps.

### Step-by-Step Forward Pass

#### Step 1: All-Gather K, V

Gather K, V from all GPUs into a buffer.

```python
# Before: local_kv shape [local_tokens, 2, heads_k, dim]
# After:  kv_buffer shape [2, total_tokens, heads_k, dim]

kv_buffer = torch.empty(
    (2, total_tokens, heads_k, head_dim),
    device=local_kv.device,
    dtype=local_kv.dtype,
)

# Prepare for all-gather
kv_for_gather = local_kv.transpose(0, 1)  # [2, local_tokens, heads_k, dim]

# All-gather across all ranks
dist.all_gather_into_tensor(kv_buffer, kv_for_gather, group=process_group)
```

**Result:** Each GPU now has complete K, V from all GPUs, but in **interleaved order**.

#### Step 2: Rearrange K, V from Interleaved to Contiguous

Convert from zigzag interleaved format to contiguous format for easier slicing.

```python
def rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size):
    """
    Input (interleaved):  [c0, c7, c1, c6, c2, c5, c3, c4]
    Output (contiguous):  [c0, c1, c2, c3, c4, c5, c6, c7]
    """
    total_tokens = kv_buffer.shape[1]
    local_tokens = total_tokens // world_size
    chunk_size = local_tokens // 2

    # Reshape to separate ranks
    kv_per_rank = kv_buffer.reshape(
        2, world_size, local_tokens, kv_buffer.shape[2], kv_buffer.shape[3]
    )

    # Extract chunks with their global indices
    chunks = []
    for rank in range(world_size):
        chunk_idx_0 = rank
        chunk_idx_1 = 2 * world_size - 1 - rank

        chunk_0 = kv_per_rank[:, rank, :chunk_size]
        chunk_1 = kv_per_rank[:, rank, chunk_size:]

        chunks.append((chunk_idx_0, chunk_0))
        chunks.append((chunk_idx_1, chunk_1))

    # Sort by chunk index and concatenate
    chunks.sort(key=lambda x: x[0])
    return torch.cat([chunk[1] for chunk in chunks], dim=1)
```

**Result:** K, V are now in contiguous order: `[c0, c1, c2, ..., c7]`

#### Step 3: Split Local Q by Global Chunk Index

Each GPU's local Q contains 2 chunks. Split them into groups based on their **global chunk index**.

```python
def split_q_by_zigzag_chunk_index(q, cu_seqlens_q, world_size, rank, n_chunks=2):
    """
    Split local Q into groups based on global chunk index.

    Example (rank=0, world_size=4):
        Local Q has: chunk[0] and chunk[7]
        chunk[0] → early group (0 < 4)
        chunk[7] → late group  (7 >= 4)
    """
    # Determine global chunk indices for this rank
    chunk_idx_0 = rank                      # First local chunk
    chunk_idx_1 = 2 * world_size - 1 - rank # Second local chunk

    # Determine which group each chunk belongs to
    group_0 = 0 if chunk_idx_0 < world_size else 1  # early or late
    group_1 = 0 if chunk_idx_1 < world_size else 1

    # Initialize output lists (one per group)
    chunk_q_list = [[] for _ in range(n_chunks)]
    chunk_cu_seqlens_q_list = [[] for _ in range(n_chunks)]
    chunk_indices_list = [[] for _ in range(n_chunks)]

    # Process each sequence
    num_sequences = len(cu_seqlens_q) - 1
    for seq_idx in range(num_sequences):
        start = cu_seqlens_q[seq_idx].item()
        end = cu_seqlens_q[seq_idx + 1].item()
        seq_len = end - start

        # Each sequence is split into n_chunks local chunks
        chunk_size = seq_len // n_chunks

        # Extract the two local chunks
        q_chunk_0 = q[start : start + chunk_size]
        q_chunk_1 = q[start + chunk_size : end]

        # Add to appropriate groups
        chunk_q_list[group_0].append(q_chunk_0)
        chunk_indices_list[group_0].append(chunk_idx_0)

        chunk_q_list[group_1].append(q_chunk_1)
        chunk_indices_list[group_1].append(chunk_idx_1)

    # Concatenate chunks within each group
    for group_idx in range(n_chunks):
        chunk_q_list[group_idx] = torch.cat(chunk_q_list[group_idx], dim=0)
        # Build cu_seqlens for this group...

    return chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list
```

**Result:**
- `chunk_q_list[0]` = all early group Q chunks
- `chunk_q_list[1]` = all late group Q chunks

#### Step 4: Compute K,V Slice Boundaries

Determine how much K,V each group needs for causal attention.

```python
def compute_kv_slices_for_groups(cu_seqlens_k, chunk_idx_0, chunk_idx_1,
                                  world_size, n_chunks=2):
    """
    Compute K,V slice end positions for each group.

    Rule:
    - Early group (chunks 0-3): needs K,V up to end of chunk 3
    - Late group (chunks 4-7):  needs K,V up to end of chunk 7
    """
    kv_slices = []

    for group_idx in range(n_chunks):
        # Determine the max chunk index in this group
        if group_idx == 0:
            max_chunk_in_group = world_size - 1  # Last early chunk
        else:
            max_chunk_in_group = 2 * world_size - 1  # Last late chunk

        # Compute end position for K,V slice
        total_seqlen = cu_seqlens_k[-1].item()
        tokens_per_chunk = total_seqlen // (2 * world_size)
        k_end_pos = (max_chunk_in_group + 1) * tokens_per_chunk

        # Create cu_seqlens_k_slice up to this position
        cu_seqlens_k_slice = cu_seqlens_k.clone()
        cu_seqlens_k_slice[cu_seqlens_k_slice > k_end_pos] = k_end_pos

        kv_slices.append((k_end_pos, cu_seqlens_k_slice))

    return kv_slices
```

**Result:**
- Early group: K,V slice ends at position after chunk 3
- Late group: K,V slice ends at position after chunk 7 (full sequence)

#### Step 5: Execute Attention (Two Modes)

**Mode A: Two-Kernels (Sequential)**

Run separate attention calls for each group.

```python
def execute_two_kernels_mode(chunk_q_list, kv_contiguous, kv_slices, ...):
    """Execute attention sequentially for each group."""
    out_chunks = []
    lse_chunks = []

    for group_idx in range(n_chunks):
        q_group = chunk_q_list[group_idx]
        k_end_pos, cu_seqlens_k_slice = kv_slices[group_idx]

        # Slice K, V for this group
        k_slice = kv_contiguous[0, :k_end_pos]  # [k_end_pos, heads_k, dim]
        v_slice = kv_contiguous[1, :k_end_pos]

        # Run flash attention
        out_group, lse_group = _flash_attn_varlen_forward(
            q_group, k_slice, v_slice,
            cu_seqlens_q_group, cu_seqlens_k_slice,
            max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax
        )

        out_chunks.append(out_group)
        lse_chunks.append(lse_group)

    return out_chunks, lse_chunks
```

**Mode B: Fused Kernel (Single Call)**

Duplicate K,V and run single attention call.

```python
def execute_fused_kernel_mode(chunk_q_list, kv_contiguous, kv_slices, ...):
    """Execute attention with fused kernel (duplicated K,V)."""

    # Concatenate all Q groups
    q_combined = torch.cat(chunk_q_list, dim=0)

    # Duplicate K,V for each group
    k_duplicated_list = []
    v_duplicated_list = []
    for k_end_pos, _ in kv_slices:
        k_duplicated_list.append(kv_contiguous[0, :k_end_pos])
        v_duplicated_list.append(kv_contiguous[1, :k_end_pos])

    k_duplicated = torch.cat(k_duplicated_list, dim=0)
    v_duplicated = torch.cat(v_duplicated_list, dim=0)

    # Prepare combined cu_seqlens
    cu_seqlens_q_combined = prepare_combined_cu_seqlens(...)
    cu_seqlens_k_combined = prepare_combined_cu_seqlens_k(...)

    # Single flash attention call
    out_combined, lse_combined = _flash_attn_varlen_forward(
        q_combined, k_duplicated, v_duplicated,
        cu_seqlens_q_combined, cu_seqlens_k_combined,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax
    )

    return out_combined, lse_combined
```

**Tradeoff:**
- Two-kernels: Simpler, less memory, slightly slower (~5%)
- Fused: Faster (~5%), more memory (duplicated K,V)

#### Step 6: Scatter Outputs Back to Original Order

The outputs are currently grouped by chunk index. Scatter them back to match the input Q order.

```python
def scatter_outputs_to_original_order(out_chunks, chunk_indices_list, original_shape):
    """
    Scatter output chunks back to original Q positions.

    Input:  [early_group_out, late_group_out]
    Output: [chunk0_out, chunk7_out] for rank 0 (in local order)
    """
    output = torch.empty(original_shape, device=out_chunks[0].device, dtype=out_chunks[0].dtype)

    for group_idx, (out_group, indices) in enumerate(zip(out_chunks, chunk_indices_list)):
        # Map each chunk back to its original position
        for local_pos, chunk_idx in enumerate(indices):
            # Extract this chunk's output
            chunk_out = extract_chunk_from_group(out_group, local_pos, ...)

            # Place back in original position
            place_chunk_at_position(output, chunk_out, local_pos, ...)

    return output
```

#### Step 7: Return Output

```python
return output, lse, chunk_info
```

The `chunk_info` contains metadata needed for backward pass:
- `chunk_indices_list`: Which chunks are in which groups
- `chunk_cu_seqlens_q_list`: Sequence boundaries within each group
- `kv_slices`: K,V slice boundaries for each group

---

## Backward Pass

### Overview

The backward pass computes gradients for Q, K, V given the gradient of the output `dout`.

**Key Challenge:** K,V regions are used by **multiple Q groups**, so gradients must be **accumulated**, not overwritten!

### Step-by-Step Backward Pass

#### Step 1: All-Gather K, V (Same as Forward)

```python
# Same as forward pass - need full K,V for gradient computation
kv_buffer = all_gather_kv(local_kv, world_size)
kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size)
```

#### Step 2: Split dout by Chunk Index

The incoming `dout` is in the same order as forward output (local order). Split it by chunk index to match the forward grouping.

```python
def split_dout_by_chunk_index(dout, chunk_indices_list, chunk_cu_seqlens_q_list):
    """Split dout into groups matching forward pass groups."""
    dout_chunks = []

    for group_idx in range(n_chunks):
        # Extract dout for this group (same order as forward)
        dout_group = extract_dout_for_group(dout, group_idx, ...)
        dout_chunks.append(dout_group)

    return dout_chunks
```

#### Step 3: Execute Backward (Two Modes)

**Mode A: Two-Kernels Backward (Sequential with Accumulation)**

```python
def backward_two_kernels_mode(dout_chunks, q, kv_contiguous, out, lse, ...):
    """
    Execute backward sequentially for each group.

    CRITICAL: Accumulate gradients for overlapping K,V regions!
    """
    # Initialize gradient buffers (FULL SIZE for K,V)
    dQ_buffer = torch.zeros_like(q)
    dK_buffer = torch.zeros_like(kv_contiguous[0])
    dV_buffer = torch.zeros_like(kv_contiguous[1])

    # Process each group
    for group_idx in range(n_chunks):
        # Get data for this group
        q_group = chunk_q_list[group_idx]
        dout_group = dout_chunks[group_idx]
        out_group = out_chunks[group_idx]
        lse_group = lse_chunks[group_idx]

        k_end_pos, cu_seqlens_k_slice = kv_slices[group_idx]
        k_slice = kv_contiguous[0, :k_end_pos]
        v_slice = kv_contiguous[1, :k_end_pos]

        # Compute gradients for this group
        dq_group, dk_slice, dv_slice = _flash_attn_varlen_backward(
            dout_group, q_group, k_slice, v_slice, out_group, lse_group,
            cu_seqlens_q_group, cu_seqlens_k_slice,
            max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic
        )

        # Scatter dQ back to original positions
        scatter_dq_to_buffer(dQ_buffer, dq_group, group_idx, ...)

        # CRITICAL: ACCUMULATE (+=) gradients for K,V, don't overwrite!
        # Multiple Q groups may use the same K,V regions
        dK_buffer[:k_end_pos] += dk_slice
        dV_buffer[:k_end_pos] += dv_slice

    return dQ_buffer, dK_buffer, dV_buffer
```

**Why Accumulation?**

```
Example (world_size=4):
- Early group (chunks 0-3) uses K,V[0:4] → produces gradients for K,V[0:4]
- Late group (chunks 4-7) uses K,V[0:8]  → produces gradients for K,V[0:8]

Overlap: K,V[0:4] receive gradients from BOTH groups!
Solution: dK_buffer[:k_end_pos] += dk_slice  (not = )
```

**Mode B: Fused Kernel Backward**

```python
def backward_fused_kernel_mode(dout_combined, q_combined, k_duplicated, v_duplicated, ...):
    """
    Execute backward with single kernel call.
    Gradients for duplicated K,V must be extracted and accumulated.
    """
    # Single backward call
    dq_combined, dk_duplicated, dv_duplicated = _flash_attn_varlen_backward(
        dout_combined, q_combined, k_duplicated, v_duplicated, out_combined, lse_combined,
        cu_seqlens_q_combined, cu_seqlens_k_combined,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic
    )

    # Extract and accumulate gradients from duplicated K,V
    dK_buffer = torch.zeros_like(kv_contiguous[0])
    dV_buffer = torch.zeros_like(kv_contiguous[1])

    offset = 0
    for group_idx, (k_end_pos, _) in enumerate(kv_slices):
        # Extract gradients for this group's K,V copy
        dk_group = dk_duplicated[offset:offset + k_end_pos]
        dv_group = dv_duplicated[offset:offset + k_end_pos]
        offset += k_end_pos

        # CRITICAL: ACCUMULATE (+=) into buffer
        dK_buffer[:k_end_pos] += dk_group
        dV_buffer[:k_end_pos] += dv_group

    # Scatter dQ back to original order
    dQ_buffer = scatter_dq_from_combined(dq_combined, chunk_indices_list, ...)

    return dQ_buffer, dK_buffer, dV_buffer
```

#### Step 4: Rearrange Gradients from Contiguous to Zigzag

The gradients dK, dV are currently in contiguous order. Convert back to zigzag interleaved format.

```python
def rearrange_grad_from_contiguous_to_zigzag(grad_contiguous, world_size):
    """
    Convert gradients from contiguous to zigzag interleaved format.

    Input (contiguous):  [c0, c1, c2, c3, c4, c5, c6, c7]
    Output (interleaved): [c0, c7, c1, c6, c2, c5, c3, c4]
    """
    total_tokens = grad_contiguous.shape[1]
    chunk_size = total_tokens // (2 * world_size)

    grad_interleaved_list = []

    for rank in range(world_size):
        chunk_idx_0 = rank
        chunk_idx_1 = 2 * world_size - 1 - rank

        # Extract the two chunks for this rank
        chunk_0 = grad_contiguous[:, chunk_idx_0 * chunk_size : (chunk_idx_0 + 1) * chunk_size]
        chunk_1 = grad_contiguous[:, chunk_idx_1 * chunk_size : (chunk_idx_1 + 1) * chunk_size]

        # Concatenate in zigzag order
        grad_interleaved_list.append(torch.cat([chunk_0, chunk_1], dim=1))

    return grad_interleaved_list
```

#### Step 5: Reduce-Scatter Gradients

Each rank needs only its local K,V gradients. Use reduce-scatter to:
1. Sum gradients across ranks
2. Scatter so each rank gets only its portion

```python
def reduce_scatter_gradients(dK_buffer, dV_buffer, world_size):
    """
    Reduce-scatter: sum gradients and distribute back to ranks.
    """
    # Rearrange to zigzag interleaved format
    dk_interleaved_list = rearrange_grad_from_contiguous_to_zigzag(dK_buffer, world_size)
    dv_interleaved_list = rearrange_grad_from_contiguous_to_zigzag(dV_buffer, world_size)

    # Stack for reduce-scatter
    dkv_interleaved = torch.stack([
        torch.cat([dk_interleaved_list[r] for r in range(world_size)], dim=1),
        torch.cat([dv_interleaved_list[r] for r in range(world_size)], dim=1),
    ], dim=0)  # [2, total_tokens, heads_k, dim]

    # Prepare output buffer (local size)
    local_dkv = torch.empty(
        (2, local_tokens, heads_k, head_dim),
        device=dkv_interleaved.device,
        dtype=dkv_interleaved.dtype,
    )

    # Reduce-scatter: sum across ranks and scatter
    dist.reduce_scatter_tensor(
        local_dkv.view(-1),
        dkv_interleaved.view(-1),
        op=dist.ReduceOp.SUM,
        group=process_group,
    )

    # Transpose back to match input format
    local_dk = local_dkv[0].transpose(0, 1)  # [local_tokens, heads_k, dim]
    local_dv = local_dkv[1].transpose(0, 1)

    return local_dk, local_dv
```

#### Step 6: Return Gradients

```python
return dQ_local, dK_local, dV_local
```

---

## Execution Modes

### Four Mode Combinations

| Mode | Forward | Backward | Memory | Speed | Complexity |
|------|---------|----------|--------|-------|------------|
| **Mode 1** | Two-Kernels | Two-Kernels | Low | Baseline | Simple |
| **Mode 2** | Two-Kernels | Fused | Medium | +2-3% | Medium |
| **Mode 3** | Fused | Two-Kernels | Medium | +2-3% | Medium |
| **Mode 4** | Fused | Fused | High | +5% | Complex |

### Detailed Memory Usage

#### Configuration for Examples
```python
# Llama3 8B configuration
world_size = 8
total_seqlen = 65536  # 8K tokens per GPU × 8 GPUs
num_heads_q = 32
num_heads_k = 8  # GQA
head_dim = 128

# Derived values
tokens_per_chunk = total_seqlen // (2 * world_size)  # 4096
tokens_early_group = tokens_per_chunk * world_size   # 32768 (chunks 0-7)
tokens_late_group = total_seqlen                      # 65536 (chunks 0-15)
```

#### Memory Breakdown Per GPU

**A. Input/Output Tensors (Same for All Modes)**

```python
# Per GPU
local_q:     local_tokens × num_heads_q × head_dim
           = 8192 × 32 × 128 = 33.6 MB (fp16)

local_kv:    local_tokens × 2 × num_heads_k × head_dim
           = 8192 × 2 × 8 × 128 = 16.8 MB (fp16)

output:      local_tokens × num_heads_q × head_dim
           = 8192 × 32 × 128 = 33.6 MB (fp16)

Total Input/Output: ~84 MB per GPU
```

**B. Intermediate Tensors - Forward Pass**

**Two-Kernels Forward:**

```python
# All-gathered K,V (contiguous format)
kv_contiguous: 2 × total_seqlen × num_heads_k × head_dim
             = 2 × 65536 × 8 × 128 = 268.4 MB

# Q groups (views, no extra memory)
q_early_group: tokens_early_group × num_heads_q × head_dim
             = 32768 × 32 × 128 = 134.2 MB (view)

q_late_group:  tokens_late_group × num_heads_q × head_dim
             = 32768 × 32 × 128 = 134.2 MB (view)

# K,V slices (views, no extra memory)
k_early_slice: tokens_early_group × num_heads_k × head_dim
v_early_slice: tokens_early_group × num_heads_k × head_dim
             = 2 × 32768 × 8 × 128 = 134.2 MB (views)

k_late_slice:  total_seqlen × num_heads_k × head_dim
v_late_slice:  total_seqlen × num_heads_k × head_dim
             = 2 × 65536 × 8 × 128 = 268.4 MB (views)

# LSE (log-sum-exp) for groups
lse_early:   tokens_early_group × num_heads_q
           = 32768 × 32 = 4.2 MB

lse_late:    tokens_late_group × num_heads_q
           = 32768 × 32 = 4.2 MB

Forward Peak Memory (Two-Kernels): 268.4 MB (kv_contiguous)
```

**Fused Forward:**

```python
# All-gathered K,V (contiguous format)
kv_contiguous: 2 × total_seqlen × num_heads_k × head_dim
             = 268.4 MB

# Q combined (concatenated)
q_combined:    2 × tokens_per_group × num_heads_q × head_dim
             = 2 × 32768 × 32 × 128 = 268.4 MB

# K,V duplicated (EXTRA MEMORY!)
k_duplicated:  (tokens_early_group + tokens_late_group) × num_heads_k × head_dim
             = (32768 + 65536) × 8 × 128 = 201.3 MB

v_duplicated:  (tokens_early_group + tokens_late_group) × num_heads_k × head_dim
             = (32768 + 65536) × 8 × 128 = 201.3 MB

# Combined LSE
lse_combined:  2 × tokens_per_group × num_heads_q
             = 2 × 32768 × 32 = 8.4 MB

Forward Peak Memory (Fused): 268.4 + 268.4 + 201.3 + 201.3 = 939.4 MB
```

**C. Intermediate Tensors - Backward Pass**

**Two-Kernels Backward:**

```python
# All-gathered K,V (recomputed or saved)
kv_contiguous: 268.4 MB

# Gradient buffers (full size)
dQ_buffer:     local_tokens × num_heads_q × head_dim
             = 8192 × 32 × 128 = 33.6 MB

dK_buffer:     total_seqlen × num_heads_k × head_dim
             = 65536 × 8 × 128 = 134.2 MB

dV_buffer:     total_seqlen × num_heads_k × head_dim
             = 65536 × 8 × 128 = 134.2 MB

# Temporary gradient slices (reused per group)
dq_group:      tokens_per_group × num_heads_q × head_dim
             = 32768 × 32 × 128 = 134.2 MB (reused)

dk_slice:      max(tokens_early, tokens_late) × num_heads_k × head_dim
             = 65536 × 8 × 128 = 134.2 MB (reused)

dv_slice:      65536 × 8 × 128 = 134.2 MB (reused)

Backward Peak Memory (Two-Kernels): 268.4 + 33.6 + 134.2 + 134.2 + 134.2 + 134.2 + 134.2
                                   = 973 MB
```

**Fused Backward:**

```python
# All-gathered K,V
kv_contiguous: 268.4 MB

# K,V duplicated (from forward, or recomputed)
k_duplicated:  201.3 MB
v_duplicated:  201.3 MB

# Q combined (from forward, or recomputed)
q_combined:    268.4 MB

# Gradient buffers
dQ_buffer:     33.6 MB
dK_buffer:     134.2 MB
dV_buffer:     134.2 MB

# Temporary gradients (single backward call)
dq_combined:   2 × tokens_per_group × num_heads_q × head_dim
             = 2 × 32768 × 32 × 128 = 268.4 MB

dk_duplicated: (tokens_early_group + tokens_late_group) × num_heads_k × head_dim
             = (32768 + 65536) × 8 × 128 = 201.3 MB

dv_duplicated: 201.3 MB

Backward Peak Memory (Fused): 268.4 + 201.3 + 201.3 + 268.4 + 33.6 + 134.2 + 134.2 + 268.4 + 201.3 + 201.3
                             = 1912 MB
```

#### Total Memory Summary Per GPU

| Mode | Forward | Backward | Total Peak | vs Baseline |
|------|---------|----------|------------|-------------|
| **Two-Kernels/Two-Kernels** | 268 MB | 973 MB | **973 MB** | 1.0× (baseline) |
| **Two-Kernels/Fused** | 268 MB | 1912 MB | **1912 MB** | 2.0× |
| **Fused/Two-Kernels** | 939 MB | 973 MB | **973 MB** | 1.0× |
| **Fused/Fused** | 939 MB | 1912 MB | **1912 MB** | 2.0× |

**Key Insight:** Backward pass dominates memory usage! Fused backward requires ~2× more memory due to duplicated K,V and their gradients.

### Memory Comparison Summary

**Two-Kernels:**
- K,V stored once in contiguous format
- Memory: `2 × total_tokens × heads_k × head_dim`
- Peak: ~973 MB per GPU (backward dominates)

**Fused:**
- K,V duplicated for each group
- Memory: `2 × (k_end_pos_group0 + k_end_pos_group1) × heads_k × head_dim`
- Peak: ~1912 MB per GPU (backward dominates)
- Approximately **2× more memory** than two-kernels

### Speed Comparison

Based on benchmarks:
- Two-Kernels: Baseline (100%)
- Fused: ~5% faster
- Reason: Single kernel launch has less overhead

### When to Use Each Mode

**Use Two-Kernels if:**
- Memory is limited
- Simplicity is preferred
- 5% speed difference doesn't matter

**Use Fused if:**
- Memory is abundant
- Maximum performance is critical
- Willing to accept complexity

---

## Load Balancing

### The Problem

In causal attention, tokens attend to different amounts of context:

```
Chunk 0: attends to [c0]                           → 1 chunk (light)
Chunk 1: attends to [c0, c1]                       → 2 chunks
Chunk 2: attends to [c0, c1, c2]                   → 3 chunks
Chunk 3: attends to [c0, c1, c2, c3]               → 4 chunks
Chunk 4: attends to [c0, c1, c2, c3, c4]           → 5 chunks
Chunk 5: attends to [c0, c1, c2, c3, c4, c5]       → 6 chunks
Chunk 6: attends to [c0, c1, c2, c3, c4, c5, c6]   → 7 chunks
Chunk 7: attends to [c0, c1, c2, c3, c4, c5, c6, c7] → 8 chunks (heavy)
```

**Contiguous Distribution (BAD):**
```
GPU 0: [c0, c1]  → 1 + 2 = 3 chunks of work
GPU 1: [c2, c3]  → 3 + 4 = 7 chunks of work
GPU 2: [c4, c5]  → 5 + 6 = 11 chunks of work
GPU 3: [c6, c7]  → 7 + 8 = 15 chunks of work  ⚠️ Imbalanced!
```

### The Solution: Zigzag Distribution

**Zigzag Distribution (GOOD):**
```
GPU 0: [c0, c7]  → 1 + 8 = 9 chunks of work
GPU 1: [c1, c6]  → 2 + 7 = 9 chunks of work
GPU 2: [c2, c5]  → 3 + 6 = 9 chunks of work
GPU 3: [c3, c4]  → 4 + 5 = 9 chunks of work  ✅ Perfect balance!
```

### Why It Works

Each GPU gets:
- One chunk from the **early group** (light work)
- One chunk from the **late group** (heavy work)
- Total work is constant: `rank + (2*world_size - 1 - rank) = 2*world_size - 1`

### Load Balancing with Grouped Attention

By splitting Q into groups and slicing K,V appropriately:

**Early group (chunks 0-3):**
- Uses K,V up to chunk 3
- 4 chunks attend to 1+2+3+4 = 10 chunks total
- Average per chunk: 2.5 chunks

**Late group (chunks 4-7):**
- Uses K,V up to chunk 7
- 4 chunks attend to 5+6+7+8 = 26 chunks total
- Average per chunk: 6.5 chunks

**Each GPU processes:**
- 1 early chunk (2.5 units of work)
- 1 late chunk (6.5 units of work)
- Total: 9 units of work ✅

---

## Computational Operations Per GPU

### Operation Count Analysis

**Configuration:**
```python
world_size = 8
total_seqlen = 65536  # 8K per GPU × 8 GPUs
tokens_per_chunk = 4096
num_heads_q = 32
num_heads_k = 8  # GQA
head_dim = 128
```

### FLOPs for Attention

Attention computation: `Q @ K^T @ V`

**FLOPs formula:**
```
FLOPs(Q_tokens, K_tokens, heads, head_dim) =
    2 × Q_tokens × K_tokens × heads × head_dim  (for Q @ K^T)
  + 2 × Q_tokens × K_tokens × heads × head_dim  (for Attn @ V)
  = 4 × Q_tokens × K_tokens × heads × head_dim
```

### Per-Chunk Operation Count

Let's calculate how many operations each chunk requires:

```
Chunk Index | Attends To | K,V Tokens | Q Tokens | Heads | FLOPs
------------|------------|------------|----------|-------|------------------
c0          | [c0]       | 4096       | 4096     | 32    | 274.9 B
c1          | [c0,c1]    | 8192       | 4096     | 32    | 549.8 B
c2          | [c0..c2]   | 12288      | 4096     | 32    | 824.6 B
c3          | [c0..c3]   | 16384      | 4096     | 32    | 1,099.5 B
c4          | [c0..c4]   | 20480      | 4096     | 32    | 1,374.4 B
c5          | [c0..c5]   | 24576      | 4096     | 32    | 1,649.3 B
c6          | [c0..c6]   | 28672      | 4096     | 32    | 1,924.1 B
c7          | [c0..c7]   | 32768      | 4096     | 32    | 2,199.0 B

Total: 9,895.6 B FLOPs across all chunks
```

**Detailed calculation for chunk 0:**
```
FLOPs = 2 × Q_tokens × K_tokens × Q_heads × head_dim  (Q @ K^T)
      + 2 × Q_tokens × K_tokens × Q_heads × head_dim  (Attn @ V)
      = 4 × Q_tokens × K_tokens × Q_heads × head_dim

FLOPs(c0) = 4 × 4096 × 4096 × 32 × 128
          = 274,877,906,944
          ≈ 274.9 B FLOPs
```

### Operations per GPU - Contiguous Distribution

```
GPU  | Chunks  | Operations (B FLOPs)         | Total FLOPs
-----|---------|------------------------------|-------------
GPU 0| [c0,c1] | 274.9 + 549.8                | 824.6 B
GPU 1| [c2,c3] | 824.6 + 1,099.5              | 1,924.1 B
GPU 2| [c4,c5] | 1,374.4 + 1,649.3            | 3,023.7 B
GPU 3| [c6,c7] | 1,924.1 + 2,199.0            | 4,123.1 B
GPU 4| [c8,c9] | 2,473.9 + 2,748.8            | 5,222.7 B
GPU 5| [c10,c11]| 3,023.7 + 3,298.5           | 6,322.2 B
GPU 6| [c12,c13]| 3,573.5 + 3,848.4           | 7,421.8 B
GPU 7| [c14,c15]| 4,123.3 + 4,398.0           | 8,521.3 B

Load imbalance: GPU 7 does 10.3× more work than GPU 0! ⚠️
```

### Operations per GPU - Zigzag Distribution

```
GPU  | Chunks   | Operations (B FLOPs)         | Total FLOPs
-----|----------|------------------------------|-------------
GPU 0| [c0,c15] | 274.9 + 4,398.0              | 4,672.9 B
GPU 1| [c1,c14] | 549.8 + 4,123.3              | 4,673.1 B
GPU 2| [c2,c13] | 824.6 + 3,848.4              | 4,673.0 B
GPU 3| [c3,c12] | 1,099.5 + 3,573.5            | 4,673.0 B
GPU 4| [c4,c11] | 1,374.4 + 3,298.5            | 4,672.9 B
GPU 5| [c5,c10] | 1,649.3 + 3,023.7            | 4,673.0 B
GPU 6| [c6,c9]  | 1,924.1 + 2,748.8            | 4,672.9 B
GPU 7| [c7,c8]  | 2,199.0 + 2,473.9            | 4,672.9 B

Perfect balance: All GPUs do ~4,673 B FLOPs! ✅
```

### Concrete Example with Real Numbers

**Setup: Llama3 8B, world_size=8, seq_len=65536**

```python
# Per chunk computation details
tokens_per_chunk = 65536 // 16 = 4096
num_heads_q = 32
head_dim = 128

def compute_flops(q_tokens, k_tokens, heads, dim):
    """Compute FLOPs for attention: 4 × Q × K × H × D"""
    return 4 * q_tokens * k_tokens * heads * dim

# Example: GPU 0 with zigzag distribution
# Chunk 0: attends to 4096 tokens (itself)
flops_c0 = compute_flops(4096, 4096, 32, 128)
# = 274,877,906,944 ≈ 274.9 GFLOPs

# Chunk 15: attends to 65536 tokens (all 16 chunks)
flops_c15 = compute_flops(4096, 65536, 32, 128)
# = 4,398,046,511,104 ≈ 4,398.0 GFLOPs

# Total for GPU 0
total_gpu0 = flops_c0 + flops_c15
# = 4,672,924,418,048 ≈ 4,672.9 GFLOPs
```

### Summary: Operations per GPU

| Distribution | Min FLOPs | Max FLOPs | Imbalance Ratio | Theoretical Impact |
|--------------|-----------|-----------|-----------------|-------------------|
| **Contiguous** | 824.6 B | 8,521.3 B | 10.3× | GPU 7 bottleneck |
| **Zigzag** | 4,672.9 B | 4,673.1 B | 1.00× | Balanced workload |

**Theoretical Analysis:**

In contiguous distribution:
- GPU 0 performs 824.6 B FLOPs (lightest workload)
- GPU 7 performs 8,521.3 B FLOPs (heaviest workload)
- **Imbalance ratio: 10.3×** - GPU 7 does 10.3× more work than GPU 0
- Bottleneck: GPU 7 determines overall throughput (other GPUs wait idle)

In zigzag distribution:
- All GPUs perform ~4,673 B FLOPs (equal workload)
- **Imbalance ratio: 1.00×** - perfect balance
- No bottleneck: All GPUs finish simultaneously
- Theoretically eliminates GPU idle time

**Hypothesis to Test:**
The 10.3× reduction in load imbalance should translate to measurable performance improvements in practice, though the actual speedup will depend on:
1. Communication overhead (all-gather, reduce-scatter)
2. Kernel launch overhead
3. Memory bandwidth constraints
4. Computation/communication overlap efficiency

This implementation provides the framework to benchmark and validate these theoretical benefits.

---

## Complete Example

### Setup

```python
world_size = 4
rank = 0  # GPU 0
sequence_length = 8192
chunk_size = 1024  # sequence_length // (2 * world_size)

# Input: Q, K, V in zigzag interleaved format
# GPU 0 has: chunk[0] + chunk[7]
local_q.shape   # [2048, 32, 128]  (2 chunks × 1024 tokens)
local_kv.shape  # [2048, 2, 8, 128]
```

### Forward Pass Example

```python
# Step 1: All-gather K,V
kv_buffer = all_gather_kv(local_kv)  # [2, 8192, 8, 128]

# Step 2: Rearrange to contiguous
kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size)
# Now: [c0, c1, c2, c3, c4, c5, c6, c7]

# Step 3: Split Q by chunk index
chunk_q_list, chunk_cu_seqlens_q_list, chunk_indices_list = \
    split_q_by_zigzag_chunk_index(local_q, cu_seqlens_q, world_size, rank)
# chunk_q_list[0] = chunk[0] (early group)
# chunk_q_list[1] = chunk[7] (late group)

# Step 4: Compute K,V slices
kv_slices = compute_kv_slices_for_groups(cu_seqlens_k, chunk_idx_0=0, chunk_idx_1=7, world_size)
# kv_slices[0] = (4096, cu_seqlens_k_slice)  # Early group: K,V up to chunk 3
# kv_slices[1] = (8192, cu_seqlens_k)        # Late group: K,V up to chunk 7

# Step 5: Execute attention (two-kernels mode)
out_chunks = []
for group_idx in [0, 1]:
    q_group = chunk_q_list[group_idx]
    k_end_pos = kv_slices[group_idx][0]

    k_slice = kv_contiguous[0, :k_end_pos]
    v_slice = kv_contiguous[1, :k_end_pos]

    out_group, lse_group = _flash_attn_varlen_forward(
        q_group, k_slice, v_slice, ...
    )
    out_chunks.append(out_group)

# Step 6: Scatter back to original order
output = scatter_outputs_to_original_order(out_chunks, chunk_indices_list, local_q.shape)
# output.shape = [2048, 32, 128]  (same as input local_q)
```

### Backward Pass Example

```python
# dout has same shape as output
dout.shape  # [2048, 32, 128]

# Step 1: All-gather K,V (same as forward)
kv_buffer = all_gather_kv(local_kv)
kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_buffer, world_size)

# Step 2: Split dout by chunk index
dout_chunks = split_dout_by_chunk_index(dout, chunk_indices_list, ...)

# Step 3: Execute backward (two-kernels mode with accumulation)
dQ_buffer = torch.zeros_like(local_q)
dK_buffer = torch.zeros_like(kv_contiguous[0])  # Full size!
dV_buffer = torch.zeros_like(kv_contiguous[1])

for group_idx in [0, 1]:
    q_group = chunk_q_list[group_idx]
    dout_group = dout_chunks[group_idx]
    k_end_pos = kv_slices[group_idx][0]

    k_slice = kv_contiguous[0, :k_end_pos]
    v_slice = kv_contiguous[1, :k_end_pos]

    dq_group, dk_slice, dv_slice = _flash_attn_varlen_backward(
        dout_group, q_group, k_slice, v_slice, ...
    )

    # Scatter dQ
    scatter_dq_to_buffer(dQ_buffer, dq_group, group_idx, ...)

    # ACCUMULATE dK, dV (critical!)
    dK_buffer[:k_end_pos] += dk_slice
    dV_buffer[:k_end_pos] += dv_slice

# Step 4: Rearrange gradients to zigzag
dk_interleaved = rearrange_grad_from_contiguous_to_zigzag(dK_buffer, world_size)
dv_interleaved = rearrange_grad_from_contiguous_to_zigzag(dV_buffer, world_size)

# Step 5: Reduce-scatter
local_dk, local_dv = reduce_scatter_gradients(dK_buffer, dV_buffer, world_size)
# local_dk.shape = [2048, 8, 128]  (same as local_kv)

# Return
return dQ_buffer, local_dk, local_dv
```

---

## Summary

### Key Takeaways

1. **Zigzag Distribution:** Each GPU gets chunks from beginning (light) and end (heavy) of sequences for perfect load balance

2. **Llama3 All-Gather:** Single-step communication for K,V, simpler than ring attention

3. **Chunk-Based Splitting:** Split Q by global chunk index (not position) into early/late groups

4. **K,V Slicing:** Early chunks use less K,V (lighter), late chunks use more K,V (heavier)

5. **Gradient Accumulation:** Critical in backward pass - overlapping K,V regions receive gradients from multiple Q groups

6. **Dual Modes:** Two-kernels (simpler, less memory) vs Fused (faster, more memory)

### Theoretical Performance Benefits

Based on the operation count analysis:

- **Perfect load balancing** across GPUs (every GPU performs equal FLOPs)
- **Simple communication** (one all-gather, one reduce-scatter per pass)
- **Flexible execution** (choose mode based on memory/speed tradeoff)
- **10.3× reduction in load imbalance** compared to contiguous distribution

**Expected Benefits (To Be Validated):**
- Elimination of GPU idle time due to balanced workload
- Potential speedup proportional to reduction in bottleneck GPU work
- Better GPU utilization in multi-GPU training scenarios

**Note:** Actual performance gains will need to be measured through benchmarking to account for:
- Communication overhead
- Memory bandwidth limitations
- Kernel launch overhead
- Real-world computation/communication overlap

### When to Consider Zigzag Llama3

✅ **Potential Use Cases:**
- Long sequences with causal attention (where load imbalance is significant)
- Multi-GPU training with 8+ GPUs
- Scenarios where GPU utilization is bottlenecked by load imbalance

❌ **Not Recommended For:**
- Short sequences (overhead may not be justified)
- Single GPU (no distribution needed)
- Non-causal attention (no load imbalance to solve)

**Important:** Benchmark your specific workload to validate if zigzag distribution provides benefits in your use case.

---

## References

- Implementation: `ring_flash_attn/zigzag_llama3_flash_attn_varlen.py`
- Test: `test/test_zigzag_llama3_flash_attn_varlen_func.py`
- Benchmark: `benchmark/benchmark_zigzag_llama3_varlen_kvpacked_func.py`
