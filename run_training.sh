#!/bin/bash
# run_training.sh
# Run Mixtral MoE training with supervised routing
# Optimized for 8x NVIDIA B200 GPUs (183GB each) with DeepSpeed ZeRO Stage 2
#
# Usage:
#   bash run_training.sh [OPTIONS]
#
# Quick Start Examples:
#   bash run_training.sh                           # 2 epochs on all data (~14-16 hours)
#   bash run_training.sh --epochs 1                # 1 epoch only (~7-8 hours)
#   bash run_training.sh --max-samples 20000       # Quick test on subset (~2-3 hours)
#   bash run_training.sh --batch-size 2            # Larger batches (if memory allows, ~20% faster)
#
# Note: Training runs with nohup by default (survives shell disconnect)
# Monitor progress: tail -f <output_dir>/training_*.log
#
# Options:
#   --routing-weight WEIGHT    Routing loss weight (default: 0.1)
#   --epochs N                 Number of epochs (default: 2)
#   --lr LR                    Learning rate (default: 1e-5)
#   --max-samples N            Limit samples for testing (default: all)
#   --batch-size N             Micro batch size per GPU (default: 1)
#   --grad-accum N             Gradient accumulation steps (default: 8)
#   --seq-length N             Max sequence length (default: 256)
#   --disable-routing          Disable supervised routing
#   --resume PATH              Resume from checkpoint
#   --output-dir DIR           Output directory (default: ./mixtral_moe_supervised)
#   --data-file FILE           Data file (default: nq_annotated_moe_balanced.jsonl)

set -e  # Exit on error

echo "========================================================================"
echo "Mixtral 8x7B MoE Training with Supervised Routing"
echo "Optimized for Multi-GPU Training with DeepSpeed ZeRO-2"
echo "========================================================================"
echo ""

# Default configuration - Optimized for 8x B200 GPUs
MODEL_ID="mistralai/Mixtral-8x7B-v0.1"
DATA_FILE="nq_annotated_moe_balanced.jsonl"
OUTPUT_DIR="./mixtral_moe_supervised"
MAX_SAMPLES=""
EPOCHS=2  # Default to 2 epochs for better convergence
MAX_SEQ_LENGTH=64
ROUTING_LOSS_WEIGHT=0.1
LEARNING_RATE=1e-5
WARMUP_STEPS=200
LOGGING_STEPS=10
SAVE_STEPS=500
PER_DEVICE_BATCH_SIZE=1  # Conservative: 1 sample per GPU
GRADIENT_ACCUMULATION_STEPS=8  # Global batch = 1 × 8 × 8 GPUs = 64
DISABLE_ROUTING=""
RESUME_FROM=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --routing-weight)
            ROUTING_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            PER_DEVICE_BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --disable-routing)
            DISABLE_ROUTING="--disable_supervised_routing"
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_training.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --routing-weight WEIGHT    Routing loss weight (default: 0.1)"
            echo "  --epochs N                 Number of epochs (default: 2)"
            echo "  --lr LR                    Learning rate (default: 1e-5)"
            echo "  --max-samples N            Limit samples for testing (default: all)"
            echo "  --batch-size N             Micro batch size per GPU (default: 1)"
            echo "  --grad-accum N             Gradient accumulation steps (default: 8)"
            echo "  --seq-length N             Max sequence length (default: 256)"
            echo "  --disable-routing          Disable supervised routing"
            echo "  --resume PATH              Resume from checkpoint"
            echo "  --output-dir DIR           Output directory (default: ./mixtral_moe_supervised)"
            echo "  --data-file FILE           Data file (default: nq_annotated_moe_balanced.jsonl)"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_training.sh --epochs 2"
            echo "  bash run_training.sh --max-samples 20000 --epochs 1"
            echo "  bash run_training.sh --batch-size 2 --grad-accum 4"
            echo ""
            echo "Note: Training runs with nohup (survives shell disconnect)"
            echo "Monitor: tail -f <output_dir>/training_*.log"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validation
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo ""
    echo "Please ensure you have the annotated dataset with expert labels."
    exit 1
fi

# Check if accelerate config exists
if [ ! -f "accelerate_config.yaml" ]; then
    echo "ERROR: accelerate_config.yaml not found"
    echo ""
    echo "Create it with: accelerate config"
    echo "Or use the existing configuration if available"
    exit 1
fi

# Auto-detect number of GPUs from accelerate config
if [ -f "accelerate_config.yaml" ]; then
    NUM_GPUS=$(grep "num_processes:" accelerate_config.yaml | awk '{print $2}')
    if [ -z "$NUM_GPUS" ]; then
        echo "WARNING: Could not detect num_processes from accelerate_config.yaml"
        NUM_GPUS=8  # Default fallback
    fi
else
    NUM_GPUS=8  # Default fallback
fi

# Count samples
TOTAL_SAMPLES=$(wc -l < "$DATA_FILE")
if [ -z "$MAX_SAMPLES" ]; then
    TRAIN_SAMPLES=$TOTAL_SAMPLES
    MAX_SAMPLES_ARG=""
else
    TRAIN_SAMPLES=$MAX_SAMPLES
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Calculate effective batch size
GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

# Extract expert parallelism from DeepSpeed config
DEEPSPEED_CONFIG="deepspeed_moe_config_stage2.json"
if [ -f "$DEEPSPEED_CONFIG" ]; then
    EXPERT_PARALLEL_SIZE=$(grep -A 10 '"moe":' "$DEEPSPEED_CONFIG" | grep "expert_parallel_size" | awk '{print $2}' | tr -d ',')
    if [ -z "$EXPERT_PARALLEL_SIZE" ]; then
        EXPERT_PARALLEL_SIZE=1  # Default fallback
    fi
else
    EXPERT_PARALLEL_SIZE=1  # Default fallback
fi
DATA_PARALLEL_SIZE=$((NUM_GPUS / EXPERT_PARALLEL_SIZE))

# Display configuration
echo "Configuration:"
echo "========================================="
echo "Model:                  $MODEL_ID"
echo "Dataset:                $DATA_FILE"
echo "Total samples:          $TOTAL_SAMPLES"
echo "Training samples:       $TRAIN_SAMPLES"
echo "Output directory:       $OUTPUT_DIR"
echo "Epochs:                 $EPOCHS"
echo "Max sequence length:    $MAX_SEQ_LENGTH"
echo "Learning rate:          $LEARNING_RATE"
echo "Warmup steps:           $WARMUP_STEPS"
echo ""
echo "Parallelization:"
echo "  Number of GPUs:       $NUM_GPUS"
echo "  Expert Parallel (EP): $EXPERT_PARALLEL_SIZE"
echo "  Data Parallel (DP):   $DATA_PARALLEL_SIZE (= $NUM_GPUS GPUs ÷ $EXPERT_PARALLEL_SIZE EP)"
echo "  Batch size per GPU:   $PER_DEVICE_BATCH_SIZE"
echo "  Gradient accum:       $GRADIENT_ACCUMULATION_STEPS"
echo "  Global batch size:    $GLOBAL_BATCH_SIZE"
echo "  (Formula: $PER_DEVICE_BATCH_SIZE × $GRADIENT_ACCUMULATION_STEPS × $NUM_GPUS)"
echo ""
if [ -z "$DISABLE_ROUTING" ]; then
    echo "Supervised Routing:     ENABLED"
    echo "  Routing loss weight:  $ROUTING_LOSS_WEIGHT"
else
    echo "Supervised Routing:     DISABLED"
fi
echo ""
if [ -n "$RESUME_FROM" ]; then
    echo "Resume from:            $RESUME_FROM"
    echo ""
fi

# Show GPU status
echo "========================================="
echo "GPU Status"
echo "========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
else
    echo "nvidia-smi not available"
fi
echo ""

# Estimate training time
# Note: The progress bar shows "iterations" (batches per device), not "steps" (weight updates)
STEPS_PER_EPOCH=$((TRAIN_SAMPLES / GLOBAL_BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))

# Iterations = batches processed per device (what you see in progress bar)
# Each GPU processes: total_samples / num_gpus batches per epoch
ITERATIONS_PER_EPOCH=$((TRAIN_SAMPLES / (PER_DEVICE_BATCH_SIZE * NUM_GPUS)))
TOTAL_ITERATIONS=$((ITERATIONS_PER_EPOCH * EPOCHS))

# Estimate iteration speed based on actual measurements (B200 + ZeRO-2 + CPU offload)
# With batch_size=1, max_seq_length=256:
# - EP=1 (expert_parallel_size=1): ~0.40 it/s (2.5 sec/iteration, recommended)
# - EP=2 (expert_parallel_size=2): ~0.33 it/s (3.0 sec/iteration, 2x more iterations)
# Using realistic estimate for EP=1: 2.5 sec/iteration
SEC_PER_ITER="2.5"  # ~0.40 it/s
SEC_PER_ITER_X100=250  # 2.5 × 100
ESTIMATED_SECONDS=$((TOTAL_ITERATIONS * SEC_PER_ITER_X100 / 100))
ESTIMATED_HOURS=$((ESTIMATED_SECONDS / 3600))

echo "Training Estimates:"
echo "========================================="
echo "Iterations per epoch:   $ITERATIONS_PER_EPOCH (batches shown in progress bar)"
echo "Weight updates/epoch:   $STEPS_PER_EPOCH (every $GRADIENT_ACCUMULATION_STEPS iterations)"
echo "Total iterations:       $TOTAL_ITERATIONS ($EPOCHS epochs)"
echo "Estimated speed:        ~${SEC_PER_ITER} sec/iteration"
echo "Estimated time:         ~${ESTIMATED_HOURS} hours"
echo "Checkpoints saved:      Every $SAVE_STEPS steps + end of each epoch"
echo ""
echo "NOTE: Actual speed depends on:"
echo "  - Expert parallelism (EP=1 is ~2x faster than EP=2 due to halved iterations)"
echo "  - Sequence length (current: $MAX_SEQ_LENGTH)"
echo "  - Batch size (larger = better GPU utilization, current: $PER_DEVICE_BATCH_SIZE)"
echo "  - CPU offload overhead (~30-40% slowdown, required for memory)"
echo "  - Network bandwidth for gradient sync"
echo ""
echo "Performance tips:"
echo "  - Increase --batch-size to 2-4 if memory allows (~20-40% faster)"
echo "  - Set expert_parallel_size=1 in DeepSpeed config (recommended for B200)"
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "========================================="
echo "Starting Training at $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================="
echo ""
echo "Training will run in background with nohup (survives shell disconnect)"
echo ""

# Save full command for reference
RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME_FROM"
fi

cat > "$OUTPUT_DIR/train_command_$(date +%Y%m%d_%H%M%S).txt" <<EOF
Training started: $(date)

Command:
accelerate launch --config_file accelerate_config.yaml \\
  train_mixtral_8x7b_moe_accelerate.py \\
  --model_id $MODEL_ID \\
  --data_file $DATA_FILE \\
  --output_dir $OUTPUT_DIR \\
  $MAX_SAMPLES_ARG \\
  --epochs $EPOCHS \\
  --max_seq_length $MAX_SEQ_LENGTH \\
  --routing_loss_weight $ROUTING_LOSS_WEIGHT \\
  --learning_rate $LEARNING_RATE \\
  --warmup_steps $WARMUP_STEPS \\
  --logging_steps $LOGGING_STEPS \\
  --save_steps $SAVE_STEPS \\
  --per_device_batch_size $PER_DEVICE_BATCH_SIZE \\
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
  $DISABLE_ROUTING \\
  $RESUME_ARG

Full Configuration:
$(cat <<CONFIG
Model: $MODEL_ID
Dataset: $DATA_FILE ($TRAIN_SAMPLES samples)
Epochs: $EPOCHS
Sequence Length: $MAX_SEQ_LENGTH
Supervised Routing: $([ -z "$DISABLE_ROUTING" ] && echo "Enabled (weight=$ROUTING_LOSS_WEIGHT)" || echo "Disabled")
Learning Rate: $LEARNING_RATE
Warmup: $WARMUP_STEPS steps

Parallelization:
  Number of GPUs: $NUM_GPUS
  Expert Parallel (EP): $EXPERT_PARALLEL_SIZE
  Data Parallel (DP): $DATA_PARALLEL_SIZE (= $NUM_GPUS GPUs ÷ $EXPERT_PARALLEL_SIZE EP)
  Batch Size per GPU: $PER_DEVICE_BATCH_SIZE
  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS steps
  Global Batch Size: $GLOBAL_BATCH_SIZE

DeepSpeed: ZeRO Stage 2 with optimizer offload
  BF16: enabled
  Activation Checkpointing: enabled
CONFIG
)
EOF

# Run training with nohup
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$OUTPUT_DIR/training.pid"

# Function to add timestamps in Pacific time to each line
add_timestamps() {
    while IFS= read -r line; do
        echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $line"
    done
}

# Create a wrapper script that will be run with nohup
WRAPPER_SCRIPT="$OUTPUT_DIR/training_wrapper.sh"
cat > "$WRAPPER_SCRIPT" <<'WRAPPER_EOF'
#!/bin/bash
add_timestamps() {
    while IFS= read -r line; do
        echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')] $line"
    done
}

accelerate launch --config_file accelerate_config.yaml \
  train_mixtral_8x7b_moe_accelerate.py \
WRAPPER_EOF

# Add the arguments to the wrapper script
cat >> "$WRAPPER_SCRIPT" <<WRAPPER_ARGS
  --model_id "$MODEL_ID" \\
  --data_file "$DATA_FILE" \\
  --output_dir "$OUTPUT_DIR" \\
  $MAX_SAMPLES_ARG \\
  --epochs $EPOCHS \\
  --max_seq_length $MAX_SEQ_LENGTH \\
  --routing_loss_weight $ROUTING_LOSS_WEIGHT \\
  --learning_rate $LEARNING_RATE \\
  --warmup_steps $WARMUP_STEPS \\
  --logging_steps $LOGGING_STEPS \\
  --save_steps $SAVE_STEPS \\
  --per_device_batch_size $PER_DEVICE_BATCH_SIZE \\
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
  $DISABLE_ROUTING \\
  $RESUME_ARG \\
  2>&1 | add_timestamps
WRAPPER_ARGS

chmod +x "$WRAPPER_SCRIPT"

# Run with nohup and capture output
nohup "$WRAPPER_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "Training started in background!"
echo "  PID: $TRAIN_PID"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if still running:"
echo "  ps -p $TRAIN_PID"
echo ""
echo "Kill training if needed:"
echo "  kill $TRAIN_PID"
echo ""
echo "Waiting for training to complete (you can safely disconnect)..."
echo ""

# Wait for the process to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Training Finished at $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================="
echo ""

# Clean up wrapper script and PID file
rm -f "$WRAPPER_SCRIPT" "$PID_FILE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo "Training log: $LOG_FILE"
    echo ""
    echo "Checkpoints:"
    find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort
    echo ""
    echo "To view training progress:"
    echo "  tensorboard --logdir $OUTPUT_DIR"
    echo ""
    echo "To evaluate the model:"
    echo "  # Load the model from checkpoint"
    echo "  # Run inference on test data"
    echo "  # Compare perplexity by expert category"
    echo ""
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo ""
    echo "Check the log file for details: $LOG_FILE"
    echo ""
    exit 1
fi
