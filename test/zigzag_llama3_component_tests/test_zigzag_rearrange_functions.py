"""
Unit tests for zigzag_llama3 rearrangement functions.

These tests verify the correctness of data rearrangement operations
that convert between zigzag interleaved and contiguous formats.
"""

import torch
import sys
sys.path.insert(0, '/workspace/ring-flash-attention')

from ring_flash_attn.zigzag_llama3_flash_attn_varlen import (
    rearrange_kv_from_zigzag_to_contiguous,
    split_q_by_zigzag_chunk_index,
)


def test_rearrange_kv_world_size_2():
    """Test K,V rearrangement for world_size=2"""
    print("=" * 70)
    print("TEST: rearrange_kv_from_zigzag_to_contiguous (world_size=2)")
    print("=" * 70)

    world_size = 2
    chunk_size = 4
    nheads_k = 2
    head_dim = 8
    total_tokens = 4 * chunk_size  # 16 tokens total

    # Create cu_seqlens for a single sequence
    cu_seqlens_tensor = torch.tensor([0, total_tokens], dtype=torch.int32)

    # Create test data where each chunk has unique values
    # Zigzag format for world_size=2:
    # GPU 0 has: chunk0 + chunk3
    # GPU 1 has: chunk1 + chunk2

    # After all-gather, we get: [GPU0_chunk0, GPU0_chunk3, GPU1_chunk1, GPU1_chunk2]
    # Expected output: [chunk0, chunk1, chunk2, chunk3]

    # Create unique values for each chunk (K and V)
    kv_zigzag = torch.zeros(2, total_tokens, nheads_k, head_dim)

    # Fill with unique values: chunk_idx * 100 + token_idx
    # GPU0's data: chunk0 (0-3) + chunk3 (12-15)
    for i in range(chunk_size):
        kv_zigzag[:, i, :, :] = 0 * 100 + i  # chunk0
    for i in range(chunk_size):
        kv_zigzag[:, chunk_size + i, :, :] = 3 * 100 + i  # chunk3

    # GPU1's data: chunk1 (4-7) + chunk2 (8-11)
    for i in range(chunk_size):
        kv_zigzag[:, 2*chunk_size + i, :, :] = 1 * 100 + i  # chunk1
    for i in range(chunk_size):
        kv_zigzag[:, 3*chunk_size + i, :, :] = 2 * 100 + i  # chunk2

    # Rearrange
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_zigzag, world_size, cu_seqlens_tensor)

    # Verify output is in order: chunk0, chunk1, chunk2, chunk3
    print(f"\nInput shape (zigzag): {kv_zigzag.shape}")
    print(f"Output shape (contiguous): {kv_contiguous.shape}")

    # Check each chunk
    success = True
    for chunk_idx in range(4):
        start = chunk_idx * chunk_size
        end = (chunk_idx + 1) * chunk_size
        expected_val = chunk_idx * 100  # Base value for this chunk
        actual_val = kv_contiguous[0, start, 0, 0].item()

        if abs(actual_val - expected_val) < 0.01:
            print(f"  ✓ Chunk {chunk_idx}: tokens [{start}:{end}] have values {expected_val}+")
        else:
            print(f"  ✗ Chunk {chunk_idx}: Expected {expected_val}, got {actual_val}")
            success = False

    # Verify all tokens in each chunk
    for chunk_idx in range(4):
        start = chunk_idx * chunk_size
        end = (chunk_idx + 1) * chunk_size
        for token in range(chunk_size):
            expected = chunk_idx * 100 + token
            actual = kv_contiguous[0, start + token, 0, 0].item()
            if abs(actual - expected) > 0.01:
                print(f"  ✗ Chunk {chunk_idx}, token {token}: Expected {expected}, got {actual}")
                success = False
                break

    if success:
        print("\n✓ TEST PASSED: Rearrangement is correct")
    else:
        print("\n✗ TEST FAILED: Rearrangement has errors")

    return success


def test_rearrange_kv_world_size_4():
    """Test K,V rearrangement for world_size=4"""
    print("\n" + "=" * 70)
    print("TEST: rearrange_kv_from_zigzag_to_contiguous (world_size=4)")
    print("=" * 70)

    world_size = 4
    chunk_size = 8
    nheads_k = 2
    head_dim = 8
    total_tokens = 8 * chunk_size  # 64 tokens total

    # Create cu_seqlens for a single sequence
    cu_seqlens_tensor = torch.tensor([0, total_tokens], dtype=torch.int32)

    # Zigzag format for world_size=4:
    # GPU 0: chunk0 + chunk7
    # GPU 1: chunk1 + chunk6
    # GPU 2: chunk2 + chunk5
    # GPU 3: chunk3 + chunk4

    # After all-gather: [GPU0_c0, GPU0_c7, GPU1_c1, GPU1_c6, GPU2_c2, GPU2_c5, GPU3_c3, GPU3_c4]
    # Expected output: [c0, c1, c2, c3, c4, c5, c6, c7]

    kv_zigzag = torch.zeros(2, total_tokens, nheads_k, head_dim)

    # Fill GPU data in zigzag order
    gpu_chunks = [
        (0, 7),  # GPU 0
        (1, 6),  # GPU 1
        (2, 5),  # GPU 2
        (3, 4),  # GPU 3
    ]

    for gpu_idx, (chunk_first, chunk_second) in enumerate(gpu_chunks):
        base_offset = gpu_idx * 2 * chunk_size
        # First chunk
        for i in range(chunk_size):
            kv_zigzag[:, base_offset + i, :, :] = chunk_first * 100 + i
        # Second chunk
        for i in range(chunk_size):
            kv_zigzag[:, base_offset + chunk_size + i, :, :] = chunk_second * 100 + i

    # Rearrange
    kv_contiguous = rearrange_kv_from_zigzag_to_contiguous(kv_zigzag, world_size, cu_seqlens_tensor)

    print(f"\nInput shape (zigzag): {kv_zigzag.shape}")
    print(f"Output shape (contiguous): {kv_contiguous.shape}")

    # Verify each chunk is in order
    success = True
    for chunk_idx in range(8):
        start = chunk_idx * chunk_size
        expected_val = chunk_idx * 100
        actual_val = kv_contiguous[0, start, 0, 0].item()

        if abs(actual_val - expected_val) < 0.01:
            print(f"  ✓ Chunk {chunk_idx} starts with value {expected_val}")
        else:
            print(f"  ✗ Chunk {chunk_idx}: Expected {expected_val}, got {actual_val}")
            success = False

    if success:
        print("\n✓ TEST PASSED")
    else:
        print("\n✗ TEST FAILED")

    return success


def test_split_q_by_zigzag_chunk_index():
    """Test Q splitting by zigzag chunk index"""
    print("\n" + "=" * 70)
    print("TEST: split_q_by_zigzag_chunk_index")
    print("=" * 70)

    world_size = 2
    rank = 0
    nheads = 4
    head_dim = 8

    # For world_size=2, rank 0 has: chunk0 + chunk3
    # With 2 groups: group0=[chunk0,chunk1], group1=[chunk2,chunk3]
    # So rank 0: chunk0 → group0, chunk3 → group1

    # Create 2 sequences, each with 16 tokens locally (8 per chunk)
    cu_seqlens_q = torch.tensor([0, 16, 32], dtype=torch.int32)
    q = torch.zeros(32, nheads, head_dim)

    # Fill with unique values: seq_idx * 1000 + chunk_type * 100 + token_idx
    # Seq 0, chunk 0 (tokens 0-7)
    for i in range(8):
        q[i, :, :] = 0 * 1000 + 0 * 100 + i
    # Seq 0, chunk 3 (tokens 8-15)
    for i in range(8):
        q[8 + i, :, :] = 0 * 1000 + 3 * 100 + i

    # Seq 1, chunk 0 (tokens 16-23)
    for i in range(8):
        q[16 + i, :, :] = 1 * 1000 + 0 * 100 + i
    # Seq 1, chunk 3 (tokens 24-31)
    for i in range(8):
        q[24 + i, :, :] = 1 * 1000 + 3 * 100 + i

    # Split Q
    chunk_q_list, chunk_cu_seqlens_list, chunk_indices_list = split_q_by_zigzag_chunk_index(
        q, cu_seqlens_q, world_size, rank, n_chunks=2
    )

    print(f"\nRank {rank} (has chunks {rank} and {2*world_size-1-rank})")
    print(f"Group 0 (early chunks): should contain chunk {rank}")
    print(f"Group 1 (late chunks): should contain chunk {2*world_size-1-rank}")

    # Verify group 0 (should have chunk 0 from both sequences)
    group0_q = chunk_q_list[0]
    print(f"\nGroup 0 shape: {group0_q.shape}")
    print(f"Group 0 cu_seqlens: {chunk_cu_seqlens_list[0]}")

    # Check first token of each sequence in group 0
    val_seq0 = group0_q[0, 0, 0].item()
    val_seq1 = group0_q[8, 0, 0].item()

    success = True
    if abs(val_seq0 - 0) < 0.01:  # Seq 0, chunk 0, token 0
        print(f"  ✓ Seq 0 in group 0 starts with 0")
    else:
        print(f"  ✗ Seq 0 in group 0: Expected 0, got {val_seq0}")
        success = False

    if abs(val_seq1 - 1000) < 0.01:  # Seq 1, chunk 0, token 0
        print(f"  ✓ Seq 1 in group 0 starts with 1000")
    else:
        print(f"  ✗ Seq 1 in group 0: Expected 1000, got {val_seq1}")
        success = False

    # Verify group 1 (should have chunk 3 from both sequences)
    group1_q = chunk_q_list[1]
    print(f"\nGroup 1 shape: {group1_q.shape}")
    print(f"Group 1 cu_seqlens: {chunk_cu_seqlens_list[1]}")

    val_seq0_g1 = group1_q[0, 0, 0].item()
    val_seq1_g1 = group1_q[8, 0, 0].item()

    if abs(val_seq0_g1 - 300) < 0.01:  # Seq 0, chunk 3, token 0
        print(f"  ✓ Seq 0 in group 1 starts with 300")
    else:
        print(f"  ✗ Seq 0 in group 1: Expected 300, got {val_seq0_g1}")
        success = False

    if abs(val_seq1_g1 - 1300) < 0.01:  # Seq 1, chunk 3, token 0
        print(f"  ✓ Seq 1 in group 1 starts with 1300")
    else:
        print(f"  ✗ Seq 1 in group 1: Expected 1300, got {val_seq1_g1}")
        success = False

    if success:
        print("\n✓ TEST PASSED")
    else:
        print("\n✗ TEST FAILED")

    return success


if __name__ == "__main__":
    print("Running unit tests for zigzag rearrangement functions\n")

    results = []
    results.append(("rearrange_kv (world_size=2)", test_rearrange_kv_world_size_2()))
    results.append(("rearrange_kv (world_size=4)", test_rearrange_kv_world_size_4()))
    results.append(("split_q_by_zigzag_chunk_index", test_split_q_by_zigzag_chunk_index()))

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
