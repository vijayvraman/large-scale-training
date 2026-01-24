#!/bin/bash
# run_mini_training.sh
# Run a mini training test with 100 samples to verify everything works
#
# Usage:
#   bash run_mini_training.sh

set -e  # Exit on error

echo "========================================================================"
echo "Mixtral MoE Mini Training Test"
echo "========================================================================"
echo ""
echo "This will run a short training test with 100 samples to verify:"
echo "  - Model loads correctly"
echo "  - Training runs without OOM errors"
echo "  - Supervised routing works"
echo "  - Logging and checkpointing work"
echo ""
echo "Expected time: ~10-15 minutes"
echo "Memory usage: ~50-60 GB per GPU"
echo ""

# Configuration
MODEL_ID="mistralai/Mixtral-8x7B-v0.1"
DATA_FILE="nq_annotated_moe.jsonl"
OUTPUT_DIR="./test_mixtral_supervised"
MAX_SAMPLES=100
EPOCHS=1
MAX_SEQ_LENGTH=64
ROUTING_LOSS_WEIGHT=0.1
LEARNING_RATE=1e-5
LOGGING_STEPS=5
SAVE_STEPS=50

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo ""
    echo "Please ensure you have the annotated dataset with expert labels."
    echo "Each line should be a JSON object with:"
    echo "  - question: string"
    echo "  - answer: string"
    echo "  - expert_label: one of [factual_lookup, numerical_reasoning, multi_hop_reasoning, commonsense_reasoning]"
    echo ""
    exit 1
fi

# Check number of samples in data file
TOTAL_SAMPLES=$(wc -l < "$DATA_FILE")
echo "Dataset: $DATA_FILE ($TOTAL_SAMPLES total samples)"
echo "Training on: $MAX_SAMPLES samples"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set logfile path
LOGFILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
echo "Logfile: $LOGFILE"
echo ""

# Show GPU status
echo "========================================="
echo "GPU Status"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo ""
echo "========================================="
echo "Starting Training"
echo "========================================="
echo ""

# Save command to file for reference
cat > "$OUTPUT_DIR/train_command.txt" <<EOF
Training started: $(date)

Command:
accelerate launch --config_file accelerate_config.yaml \\
  train_mixtral_8x7b_moe_accelerate.py \\
  --model_id $MODEL_ID \\
  --data_file $DATA_FILE \\
  --output_dir $OUTPUT_DIR \\
  --max_samples $MAX_SAMPLES \\
  --epochs $EPOCHS \\
  --max_seq_length $MAX_SEQ_LENGTH \\
  --routing_loss_weight $ROUTING_LOSS_WEIGHT \\
  --learning_rate $LEARNING_RATE \\
  --logging_steps $LOGGING_STEPS \\
  --save_steps $SAVE_STEPS \\
  --warmup_steps 20 \\
  --per_device_batch_size 1 \\
  --gradient_accumulation_steps 8

Configuration:
- Model: $MODEL_ID
- Samples: $MAX_SAMPLES
- Epochs: $EPOCHS
- Sequence length: $MAX_SEQ_LENGTH
- Routing loss weight: $ROUTING_LOSS_WEIGHT
- Learning rate: $LEARNING_RATE
EOF

# Run training (output to both screen and logfile)
accelerate launch --config_file accelerate_config.yaml \
  train_mixtral_8x7b_moe_accelerate.py \
  --model_id "$MODEL_ID" \
  --data_file "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples $MAX_SAMPLES \
  --epochs $EPOCHS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --routing_loss_weight $ROUTING_LOSS_WEIGHT \
  --learning_rate $LEARNING_RATE \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  --warmup_steps 20 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 8 2>&1 | tee "$LOGFILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
echo "Training Complete"
echo "========================================="
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Mini training completed successfully!"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo "Training log: $LOGFILE"
    echo ""

    # Show what was created
    echo "Files created:"
    ls -lh "$OUTPUT_DIR" | tail -n +2

    echo ""
    echo "To view training logs in TensorBoard:"
    echo "  tensorboard --logdir $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Check the logs to verify loss is decreasing"
    echo "  2. Verify routing supervision loss is logged"
    echo "  3. Run full training: bash run_training.sh"
    echo ""
    exit 0
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo ""
    echo "Check the logs above for error messages."
    echo "Common issues:"
    echo "  - Out of memory: Reduce max_seq_length or batch size"
    echo "  - Model download failed: Check internet connection"
    echo "  - CUDA errors: Check GPU drivers"
    echo ""
    exit 1
fi
