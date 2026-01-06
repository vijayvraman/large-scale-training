#!/bin/bash
# run_tests.sh
# Run all unit tests to verify the implementation before training
#
# Usage:
#   bash run_tests.sh

set -e  # Exit on error

echo "========================================================================"
echo "Running Mixtral MoE Implementation Tests"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Track overall success
TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Check Python environment
echo "========================================="
echo "Test 0: Environment Check"
echo "========================================="
echo ""

echo "Checking Python version..."
python --version
if [ $? -eq 0 ]; then
    print_status 0 "Python available"
else
    print_status 1 "Python not found"
    exit 1
fi

echo ""
echo "Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status 0 "PyTorch installed"
else
    print_status 1 "PyTorch not installed"
    echo "Install with: pip install torch"
    exit 1
fi

python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status 0 "Transformers installed"
else
    print_status 1 "Transformers not installed"
    echo "Install with: pip install transformers"
    exit 1
fi

python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status 0 "Accelerate installed"
else
    print_status 1 "Accelerate not installed"
    echo "Install with: pip install accelerate"
    exit 1
fi

python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status 0 "DeepSpeed installed"
else
    print_status 1 "DeepSpeed not installed"
    echo "Install with: pip install deepspeed"
    exit 1
fi

echo ""
echo "========================================="
echo "Test 1: Supervised Routing Module"
echo "========================================="
echo ""

python test_supervised_routing.py
TEST1_STATUS=$?

if [ $TEST1_STATUS -eq 0 ]; then
    print_status 0 "Supervised routing tests PASSED"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    print_status 1 "Supervised routing tests FAILED"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo "========================================="
echo "Test 2: Model Loading Configuration"
echo "========================================="
echo ""

python test_model_loading.py
TEST2_STATUS=$?

if [ $TEST2_STATUS -eq 0 ]; then
    print_status 0 "Model loading tests PASSED"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    print_status 1 "Model loading tests FAILED"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo "========================================="
echo "Test 3: Configuration Files"
echo "========================================="
echo ""

# Check if required files exist
echo "Checking configuration files..."

if [ -f "deepspeed_moe_config.json" ]; then
    print_status 0 "deepspeed_moe_config.json exists"

    # Validate JSON
    python -c "import json; json.load(open('deepspeed_moe_config.json'))" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_status 0 "deepspeed_moe_config.json is valid JSON"

        # Check key MoE settings
        NUM_EXPERTS=$(python -c "import json; config=json.load(open('deepspeed_moe_config.json')); print(config['moe']['num_experts'])")
        TOP_K=$(python -c "import json; config=json.load(open('deepspeed_moe_config.json')); print(config['moe']['top_k'])")

        echo "  - num_experts: $NUM_EXPERTS (expected: 8)"
        echo "  - top_k: $TOP_K (expected: 2)"

        if [ "$NUM_EXPERTS" = "8" ] && [ "$TOP_K" = "2" ]; then
            print_status 0 "MoE configuration correct"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            print_status 1 "MoE configuration incorrect"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        print_status 1 "deepspeed_moe_config.json is invalid JSON"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    print_status 1 "deepspeed_moe_config.json not found"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

if [ -f "accelerate_config.yaml" ]; then
    print_status 0 "accelerate_config.yaml exists"
else
    print_status 1 "accelerate_config.yaml not found"
    echo "  Note: Create with 'accelerate config' command"
fi

if [ -f "supervised_routing.py" ]; then
    print_status 0 "supervised_routing.py exists"
else
    print_status 1 "supervised_routing.py not found"
fi

if [ -f "train_mixtral_8x7b_moe_accelerate.py" ]; then
    print_status 0 "train_mixtral_8x7b_moe_accelerate.py exists"
else
    print_status 1 "train_mixtral_8x7b_moe_accelerate.py not found"
fi

echo ""
echo "========================================="
echo "Test 4: GPU Availability"
echo "========================================="
echo ""

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null

if [ $? -eq 0 ]; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    if [ "$GPU_COUNT" -ge 2 ]; then
        print_status 0 "Multiple GPUs available ($GPU_COUNT GPUs)"
    elif [ "$GPU_COUNT" -eq 1 ]; then
        echo -e "${YELLOW}⚠ Only 1 GPU available (training will work but won't use expert parallelism)${NC}"
    else
        echo -e "${YELLOW}⚠ No GPUs available (training will be very slow on CPU)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Could not check GPU availability${NC}"
fi

echo ""
echo "========================================================================"
echo "Test Results Summary"
echo "========================================================================"
echo ""
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}ALL TESTS PASSED! ✓${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Prepare your dataset (nq_annotated_moe.jsonl)"
    echo "  2. Run mini training test: bash run_mini_training.sh"
    echo "  3. Run full training: bash run_training.sh"
    echo ""
    exit 0
else
    echo -e "${RED}SOME TESTS FAILED ✗${NC}"
    echo ""
    echo "Please fix the issues above before proceeding with training."
    echo ""
    exit 1
fi
