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

# Check for DeepSpeed config files (stage 2 and stage 3)
FOUND_DEEPSPEED_CONFIG=false
for config_file in "deepspeed_moe_config_stage2.json" "deepspeed_moe_config_stage3.json"; do
    if [ -f "$config_file" ]; then
        print_status 0 "$config_file exists"
        FOUND_DEEPSPEED_CONFIG=true

        # Validate JSON
        python -c "import json; json.load(open('$config_file'))" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_status 0 "$config_file is valid JSON"

            # Check key MoE settings
            NUM_EXPERTS=$(python -c "import json; config=json.load(open('$config_file')); print(config['moe']['num_experts'])")
            TOP_K=$(python -c "import json; config=json.load(open('$config_file')); print(config['moe']['top_k'])")
            ZERO_STAGE=$(python -c "import json; config=json.load(open('$config_file')); print(config['zero_optimization']['stage'])")

            echo "  - num_experts: $NUM_EXPERTS (expected: 8)"
            echo "  - top_k: $TOP_K (expected: 2)"
            echo "  - ZeRO stage: $ZERO_STAGE"

            if [ "$NUM_EXPERTS" = "8" ] && [ "$TOP_K" = "2" ]; then
                print_status 0 "MoE configuration correct in $config_file"
            else
                print_status 1 "MoE configuration incorrect in $config_file"
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
        else
            print_status 1 "$config_file is invalid JSON"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    fi
done

if [ "$FOUND_DEEPSPEED_CONFIG" = true ]; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    print_status 1 "No DeepSpeed config files found (expected deepspeed_moe_config_stage2.json or deepspeed_moe_config_stage3.json)"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

if [ -f "accelerate_config.yaml" ]; then
    print_status 0 "accelerate_config.yaml exists"

    # Update accelerate_config.yaml with detected GPU count
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -gt 0 ]; then
        sed -i "s/^num_processes: .*/num_processes: $NUM_GPUS/" accelerate_config.yaml
        print_status 0 "Updated accelerate_config.yaml to use $NUM_GPUS processes"

        # Detect GPU memory and choose appropriate DeepSpeed config
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        TOTAL_GPU_MEMORY=$((NUM_GPUS * GPU_MEMORY))

        # For Mixtral-8x7B (46.7B params ~93GB in bf16):
        # - Stage 2: Full model on each GPU (needs ~100GB+ per GPU)
        # - Stage 3: Model sharded across GPUs + CPU offload (works with 2x80GB)
        # Use stage 3 if we have less than 100GB per GPU or total memory < 300GB
        if [ "$GPU_MEMORY" -lt 100000 ] || [ "$TOTAL_GPU_MEMORY" -lt 300000 ]; then
            DEEPSPEED_CONFIG="deepspeed_moe_config_stage3.json"
            ZERO3_FLAG="true"
            echo "  - Detected ${NUM_GPUS}x GPUs with ${GPU_MEMORY}MB each (total: ${TOTAL_GPU_MEMORY}MB)"
            echo "  - Using ZeRO Stage 3 with CPU offloading for large model support"
        else
            DEEPSPEED_CONFIG="deepspeed_moe_config_stage2.json"
            ZERO3_FLAG="false"
            echo "  - Detected ${NUM_GPUS}x GPUs with ${GPU_MEMORY}MB each (total: ${TOTAL_GPU_MEMORY}MB)"
            echo "  - Using ZeRO Stage 2 (sufficient GPU memory available)"
        fi

        # Update accelerate config with appropriate DeepSpeed config
        sed -i "s|deepspeed_config_file: .*|deepspeed_config_file: $DEEPSPEED_CONFIG|" accelerate_config.yaml
        sed -i "s/zero3_init_flag: .*/zero3_init_flag: $ZERO3_FLAG/" accelerate_config.yaml
        print_status 0 "Updated accelerate_config.yaml to use $DEEPSPEED_CONFIG"
    else
        echo -e "${YELLOW}⚠ Could not detect GPUs with nvidia-smi${NC}"
    fi
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
