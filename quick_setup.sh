#!/bin/bash
# Quick Setup Script for Grouped Flash Attention Testing
# Run this on your remote GPU to get started quickly

set -e  # Exit on error

echo "=========================================="
echo "Grouped Flash Attention - Quick Setup"
echo "=========================================="
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. This requires an NVIDIA GPU."
    exit 1
fi

echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install torch triton ninja packaging pytest --quiet

# Install flash-attention (official, for baseline)
echo "📦 Installing flash-attention (5-10 minutes)..."
pip install flash-attn --no-build-isolation --quiet || {
    echo "⚠️  Warning: flash-attn installation failed, continuing anyway..."
}

# Clone ring-flash-attention
echo "📥 Cloning ring-flash-attention..."
if [ -d "ring-flash-attention" ]; then
    echo "   Directory exists, pulling latest..."
    cd ring-flash-attention
    git fetch origin
    git checkout feature/grouped-flash-attention
    git pull origin feature/grouped-flash-attention
else
    git clone -b feature/grouped-flash-attention https://github.com/petrpan26/ring-flash-attention.git
    cd ring-flash-attention
fi

# Install ring-flash-attention
echo "📦 Installing ring-flash-attention..."
pip install -e . --quiet

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Test:"
echo "  python example_grouped_attention.py"
echo ""
echo "Run Tests:"
echo "  pytest test/test_grouped_flash_attention.py -v"
echo ""
echo "Benchmarks:"
echo "  python benchmark/benchmark_grouped_attention.py"
echo ""
echo "For detailed instructions, see:"
echo "  - claude.md (testing guide)"
echo "  - SETUP_GUIDE.md (complete setup options)"
echo ""
