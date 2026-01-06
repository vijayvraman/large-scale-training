#!/bin/bash
# run_training.sh
# Run full Mixtral MoE training with supervised routing
#
# Usage:
#   bash run_training.sh [OPTIONS]
#
# Options:
#   --routing-weight WEIGHT    Routing loss weight (default: 0.1)
#   --epochs N                 Number of epochs (default: 3)
#   --lr LR                    Learning rate (default: 1e-5)
#   --max-samples N            Limit samples for testing (default: all)
#   --disable-routing          Disable supervised routing
#   --resume PATH              Resume from checkpoint

set -e  # Exit on error

echo "========================================================================"
echo "Mixtral MoE Full Training with Supervised Routing"
echo "========================================================================"
echo ""

# Default configuration
MODEL_ID="mistralai/Mixtral-8x7B-v0.1"
DATA_FILE="nq_annotated_moe.jsonl"
OUTPUT_DIR="./mixtral_moe_supervised"
MAX_SAMPLES=""
EPOCHS=3
MAX_SEQ_LENGTH=256
ROUTING_LOSS_WEIGHT=0.1
LEARNING_RATE=1e-5
WARMUP_STEPS=200
LOGGING_STEPS=10
SAVE_STEPS=500
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
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
            echo "  --epochs N                 Number of epochs (default: 3)"
            echo "  --lr LR                    Learning rate (default: 1e-5)"
            echo "  --max-samples N            Limit samples for testing (default: all)"
            echo "  --disable-routing          Disable supervised routing"
            echo "  --resume PATH              Resume from checkpoint"
            echo "  --output-dir DIR           Output directory (default: ./mixtral_moe_supervised)"
            echo "  --data-file FILE           Data file (default: nq_annotated_moe.jsonl)"
            echo "  --help                     Show this help message"
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
echo "Batch size per GPU:     $PER_DEVICE_BATCH_SIZE"
echo "Gradient accum steps:   $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size:   $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2))"
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
STEPS_PER_EPOCH=$((TRAIN_SAMPLES / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2)))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
ESTIMATED_HOURS=$((TOTAL_STEPS / 360))  # Assuming ~0.5 it/s

echo "Training Estimates:"
echo "========================================="
echo "Steps per epoch:        ~$STEPS_PER_EPOCH"
echo "Total training steps:   ~$TOTAL_STEPS"
echo "Estimated time:         ~$ESTIMATED_HOURS hours"
echo "Checkpoints saved:      Every $SAVE_STEPS steps + end of each epoch"
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
echo "Starting Training at $(date)"
echo "========================================="
echo ""

# Save full command for reference
RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME_FROM"
fi

cat > "$OUTPUT_DIR/train_command.txt" <<EOF
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
Supervised Routing: $([ -z "$DISABLE_ROUTING" ] && echo "Enabled (weight=$ROUTING_LOSS_WEIGHT)" || echo "Disabled")
Learning Rate: $LEARNING_RATE
Warmup: $WARMUP_STEPS steps
Batch Size: $PER_DEVICE_BATCH_SIZE per GPU
Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS steps
Effective Batch Size: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2))
CONFIG
)
EOF

# Run training with tee to save output
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

accelerate launch --config_file accelerate_config.yaml \
  train_mixtral_8x7b_moe_accelerate.py \
  --model_id "$MODEL_ID" \
  --data_file "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  $MAX_SAMPLES_ARG \
  --epochs $EPOCHS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --routing_loss_weight $ROUTING_LOSS_WEIGHT \
  --learning_rate $LEARNING_RATE \
  --warmup_steps $WARMUP_STEPS \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  $DISABLE_ROUTING \
  $RESUME_ARG \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
echo "Training Finished at $(date)"
echo "========================================="
echo ""

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
