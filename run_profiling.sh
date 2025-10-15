#!/bin/bash
# Quick profiling script for Zigzag Ring vs Llama3 comparison

set -e

echo "======================================================================"
echo "GPU Profiling: Zigzag Ring vs Zigzag Llama3"
echo "======================================================================"
echo ""

# Configuration
NPROC=${NPROC:-4}  # Number of GPUs
SEQLEN=${SEQLEN:-2048}
BATCH=${BATCH:-2}
ITERATIONS=${ITERATIONS:-20}

echo "Configuration:"
echo "  GPUs: $NPROC"
echo "  Sequence Length: $SEQLEN"
echo "  Batch Size: $BATCH"
echo "  Iterations: $ITERATIONS"
echo ""

# Create output directory
mkdir -p profiler_logs
mkdir -p flamegraphs

# Method 1: PyTorch Profiler (Default)
echo "======================================================================"
echo "METHOD 1: PyTorch Profiler (TensorBoard + Flamegraphs)"
echo "======================================================================"
echo ""

echo "Running profiling..."
torchrun --nproc_per_node=$NPROC benchmark_with_profiling.py \
    --mode pytorch \
    --seqlen $SEQLEN \
    --batch $BATCH \
    --nheads 32 \
    --d 128 \
    --iterations $ITERATIONS

echo ""
echo "Profiling complete! Results in ./profiler_logs/"
echo ""

# Generate flamegraphs if flamegraph.pl is available
if command -v flamegraph.pl &> /dev/null; then
    echo "Generating flamegraphs..."

    if [ -f profiler_logs/zigzag_ring_stacks_rank0.txt ]; then
        flamegraph.pl profiler_logs/zigzag_ring_stacks_rank0.txt > flamegraphs/zigzag_ring_flamegraph.svg
        echo "  ✓ Ring flamegraph: flamegraphs/zigzag_ring_flamegraph.svg"
    fi

    if [ -f profiler_logs/zigzag_llama3_stacks_rank0.txt ]; then
        flamegraph.pl profiler_logs/zigzag_llama3_stacks_rank0.txt > flamegraphs/zigzag_llama3_flamegraph.svg
        echo "  ✓ Llama3 flamegraph: flamegraphs/zigzag_llama3_flamegraph.svg"
    fi

    echo ""
else
    echo "⚠ flamegraph.pl not found. Install FlameGraph for flamegraph visualization:"
    echo "  git clone https://github.com/brendangregg/FlameGraph"
    echo "  export PATH=\$PATH:\$PWD/FlameGraph"
    echo ""
fi

echo "======================================================================"
echo "VISUALIZATION OPTIONS"
echo "======================================================================"
echo ""
echo "1. TensorBoard (Interactive Timeline):"
echo "   tensorboard --logdir=./profiler_logs"
echo "   Then open: http://localhost:6006"
echo ""
echo "2. Chrome Trace (Detailed Timeline):"
echo "   Open chrome://tracing in Chrome"
echo "   Load: profiler_logs/zigzag_ring_trace_rank0.json"
echo "   Load: profiler_logs/zigzag_llama3_trace_rank0.json"
echo ""
echo "3. Flamegraphs (Call Stack Analysis):"
echo "   Open in browser:"
echo "   - flamegraphs/zigzag_ring_flamegraph.svg"
echo "   - flamegraphs/zigzag_llama3_flamegraph.svg"
echo ""

echo "======================================================================"
echo "OPTIONAL: Nsight Systems (Professional)"
echo "======================================================================"
echo ""
echo "For more detailed analysis, run with Nsight Systems:"
echo ""
echo "  nsys profile -o zigzag_comparison \\"
echo "    --trace=cuda,nvtx,osrt \\"
echo "    --capture-range=cudaProfilerApi \\"
echo "    torchrun --nproc_per_node=$NPROC benchmark_with_profiling.py --mode nsight"
echo ""
echo "  Then open: nsys-ui zigzag_comparison.nsys-rep"
echo ""

echo "======================================================================"
echo "COMPARISON CHECKLIST"
echo "======================================================================"
echo ""
echo "Compare these metrics between Ring and Llama3:"
echo ""
echo "  [ ] GPU Utilization (Ring should be ~95%, Llama3 ~75-85%)"
echo "  [ ] Communication overlap (Ring should overlap, Llama3 sequential)"
echo "  [ ] Kernel count (Llama3 should have extra Triton kernels)"
echo "  [ ] Idle gaps (Llama3 should have gap during all-gather)"
echo "  [ ] Total time (Ring should be ~12% faster)"
echo ""
echo "See PROFILING_GUIDE.md for detailed analysis instructions."
echo ""
