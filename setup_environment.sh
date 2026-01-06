#!/bin/bash
# setup_environment.sh
# Set up the environment for Mixtral MoE training
#
# Usage:
#   bash setup_environment.sh

set -e

echo "========================================================================"
echo "Mixtral MoE Training Environment Setup"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3.8+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo ""
    read -p "Create virtual environment? (recommended) (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -m venv .venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
        echo ""
        echo "Activate it with: source .venv/bin/activate"
        echo ""
    fi
fi

# Install packages
echo "Installing required packages..."
echo ""

PACKAGES=(
    "torch>=2.0.0"
    "transformers>=4.36.0"
    "accelerate>=0.26.0"
    "deepspeed>=0.12.0"
    "tensorboard"
    "tqdm"
    "numpy"
    "datasets"
)

for package in "${PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install -q "$package" || echo -e "${YELLOW}⚠ Failed to install $package${NC}"
done

echo ""
echo -e "${GREEN}✓ Package installation complete${NC}"
echo ""

# Verify installations
echo "Verifying installations..."
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"

echo ""

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""

# Configure Accelerate if not already configured
if [ ! -f "accelerate_config.yaml" ]; then
    echo "========================================="
    echo "Accelerate Configuration"
    echo "========================================="
    echo ""
    echo "Accelerate needs to be configured for distributed training."
    echo ""
    read -p "Run accelerate config now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Answer the following questions:"
        echo "  - Compute environment: This machine"
        echo "  - Distributed type: DeepSpeed"
        echo "  - DeepSpeed config: deepspeed_moe_config.json"
        echo "  - Number of GPUs: 2 (or your GPU count)"
        echo "  - Use FP16/BF16: Yes (BF16 recommended)"
        echo ""
        accelerate config
    else
        echo ""
        echo "You can configure accelerate later with: accelerate config"
        echo ""
    fi
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x run_tests.sh run_mini_training.sh run_training.sh monitor_training.sh 2>/dev/null || true
echo -e "${GREEN}✓ Scripts are executable${NC}"
echo ""

# Check for dataset
echo "Checking for dataset..."
if [ -f "nq_annotated_moe.jsonl" ]; then
    SAMPLE_COUNT=$(wc -l < nq_annotated_moe.jsonl)
    echo -e "${GREEN}✓ Dataset found: nq_annotated_moe.jsonl ($SAMPLE_COUNT samples)${NC}"
else
    echo -e "${YELLOW}⚠ Dataset not found: nq_annotated_moe.jsonl${NC}"
    echo ""
    echo "Your dataset should be a JSONL file with:"
    echo "  {\"question\": \"...\", \"answer\": \"...\", \"expert_label\": \"factual_lookup\"}"
    echo ""
    echo "Expert labels should be one of:"
    echo "  - factual_lookup"
    echo "  - numerical_reasoning"
    echo "  - multi_hop_reasoning"
    echo "  - commonsense_reasoning"
    echo ""
fi

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate virtual environment (if created):"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run tests to verify setup:"
echo "     bash run_tests.sh"
echo ""
echo "  3. Try mini training (10-15 min):"
echo "     bash run_mini_training.sh"
echo ""
echo "  4. Start full training:"
echo "     bash run_training.sh"
echo ""
echo "  5. Monitor training (in another terminal):"
echo "     bash monitor_training.sh"
echo ""
echo "For detailed instructions, see: README.md"
echo ""
