"""
Test the extract_local function used in zigzag_llama3 test.

This function extracts the local zigzag-distributed portion for each rank.
"""

import torch
import sys
sys.path.insert(0, '/workspace/ring-flash-attention/test')


def extract_local(value, cu_seqlens, rank, world_size):
    """Extract local zigzag-distributed portion for this rank.

    Each sequence is split into 2*world_size chunks.
    GPU gets: chunk[rank] + chunk[2*world_size - 1 - rank]

    This creates the interleaved zigzag pattern where each GPU
    gets tokens from both beginning and end of sequences.
    """
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend([
            local_value[rank].detach().clone(),
            local_value[2 * world_size - 1 - rank].detach().clone(),
        ])
    return torch.cat(local_values, dim=0).contiguous()


def test_extract_local():
    """Test extract_local function"""
    print("=" * 70)
    print("TEST: extract_local function")
    print("=" * 70)

    world_size = 2
    # Create 1 sequence with 16 tokens (divisible by 2*world_size=4)
    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)

    # Create test data with unique values
    # Tokens 0-15, each has value equal to its index
    value = torch.arange(16).unsqueeze(-1).unsqueeze(-1).float()  # [16, 1, 1]

    print(f"\nSequence: {value.squeeze().tolist()}")
    print(f"Total tokens: 16")
    print(f"Chunks (2*world_size=4): 4 chunks of 4 tokens each")
    print(f"  Chunk 0: tokens [0,1,2,3]")
    print(f"  Chunk 1: tokens [4,5,6,7]")
    print(f"  Chunk 2: tokens [8,9,10,11]")
    print(f"  Chunk 3: tokens [12,13,14,15]")

    # Test rank 0
    print(f"\nRank 0 should get: chunk[0] + chunk[3]")
    print(f"  = tokens [0,1,2,3] + [12,13,14,15]")

    local_rank0 = extract_local(value, cu_seqlens, rank=0, world_size=world_size)
    expected_rank0 = torch.tensor([0,1,2,3, 12,13,14,15], dtype=torch.float32)

    print(f"  Extracted: {local_rank0.squeeze().tolist()}")
    print(f"  Expected:  {expected_rank0.tolist()}")

    if torch.allclose(local_rank0.squeeze(), expected_rank0):
        print(f"  ✓ Rank 0 extraction CORRECT")
        rank0_pass = True
    else:
        print(f"  ✗ Rank 0 extraction WRONG")
        rank0_pass = False

    # Test rank 1
    print(f"\nRank 1 should get: chunk[1] + chunk[2]")
    print(f"  = tokens [4,5,6,7] + [8,9,10,11]")

    local_rank1 = extract_local(value, cu_seqlens, rank=1, world_size=world_size)
    expected_rank1 = torch.tensor([4,5,6,7, 8,9,10,11], dtype=torch.float32)

    print(f"  Extracted: {local_rank1.squeeze().tolist()}")
    print(f"  Expected:  {expected_rank1.tolist()}")

    if torch.allclose(local_rank1.squeeze(), expected_rank1):
        print(f"  ✓ Rank 1 extraction CORRECT")
        rank1_pass = True
    else:
        print(f"  ✗ Rank 1 extraction WRONG")
        rank1_pass = False

    # Test with multiple sequences
    print(f"\n" + "=" * 70)
    print("TEST: Multiple sequences")
    print("=" * 70)

    cu_seqlens_multi = torch.tensor([0, 8, 16], dtype=torch.int32)
    value_multi = torch.arange(16).unsqueeze(-1).unsqueeze(-1).float()

    print(f"\n2 sequences: [0-7] and [8-15]")
    print(f"Seq 0 (8 tokens) → 4 chunks of 2 tokens")
    print(f"  Chunk 0: [0,1], Chunk 1: [2,3], Chunk 2: [4,5], Chunk 3: [6,7]")
    print(f"Seq 1 (8 tokens) → 4 chunks of 2 tokens")
    print(f"  Chunk 0: [8,9], Chunk 1: [10,11], Chunk 2: [12,13], Chunk 3: [14,15]")

    print(f"\nRank 0 should get:")
    print(f"  Seq0: chunk[0]+chunk[3] = [0,1]+[6,7]")
    print(f"  Seq1: chunk[0]+chunk[3] = [8,9]+[14,15]")
    print(f"  Total: [0,1,6,7, 8,9,14,15]")

    local_rank0_multi = extract_local(value_multi, cu_seqlens_multi, rank=0, world_size=world_size)
    expected_rank0_multi = torch.tensor([0,1,6,7, 8,9,14,15], dtype=torch.float32)

    print(f"  Extracted: {local_rank0_multi.squeeze().tolist()}")
    print(f"  Expected:  {expected_rank0_multi.tolist()}")

    if torch.allclose(local_rank0_multi.squeeze(), expected_rank0_multi):
        print(f"  ✓ Multi-sequence extraction CORRECT")
        multi_pass = True
    else:
        print(f"  ✗ Multi-sequence extraction WRONG")
        multi_pass = False

    # Final result
    all_pass = rank0_pass and rank1_pass and multi_pass
    print(f"\n" + "=" * 70)
    if all_pass:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = test_extract_local()
    sys.exit(0 if success else 1)
