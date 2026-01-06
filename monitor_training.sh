#!/bin/bash
# monitor_training.sh
# Monitor ongoing training with real-time metrics
#
# Usage:
#   bash monitor_training.sh [OUTPUT_DIR]

OUTPUT_DIR="${1:-./mixtral_moe_supervised}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo "========================================================================"
echo "Mixtral MoE Training Monitor"
echo "========================================================================"
echo ""

# Function to display section header
section_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to refresh display
monitor_loop() {
    while true; do
        clear
        echo "========================================================================"
        echo "Mixtral MoE Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
        echo ""

        # GPU Status
        section_header "GPU Status"
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU %s: %s\n  Temp: %s°C | GPU Util: %s%% | Mem Util: %s%% | Mem: %s/%s MB\n\n", $1, $2, $3, $4, $5, $6, $7}'
        else
            echo "nvidia-smi not available"
        fi
        echo ""

        # Check if training is running
        section_header "Training Status"
        if pgrep -f "train_mixtral_8x7b_moe_accelerate.py" > /dev/null; then
            echo -e "${GREEN}✓ Training is RUNNING${NC}"
            TRAINING_PID=$(pgrep -f "train_mixtral_8x7b_moe_accelerate.py" | head -1)
            echo "PID: $TRAINING_PID"

            # Show process info
            ps -p $TRAINING_PID -o pid,etime,pcpu,pmem,cmd | tail -1
        else
            echo -e "${YELLOW}⚠ Training is NOT running${NC}"
        fi
        echo ""

        # Recent logs
        section_header "Recent Training Logs (Last 10 lines)"
        if [ -d "$OUTPUT_DIR" ]; then
            # Find most recent log file
            LATEST_LOG=$(find "$OUTPUT_DIR" -name "training_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

            if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
                tail -10 "$LATEST_LOG" | grep -E "(Step|Loss|Epoch|checkpoint)" || echo "No recent training logs found"
            else
                echo "No log files found in $OUTPUT_DIR"
            fi
        else
            echo "Output directory not found: $OUTPUT_DIR"
        fi
        echo ""

        # Checkpoint info
        section_header "Checkpoints"
        if [ -d "$OUTPUT_DIR" ]; then
            CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | wc -l)
            echo "Total checkpoints: $CHECKPOINT_COUNT"

            if [ $CHECKPOINT_COUNT -gt 0 ]; then
                echo ""
                echo "Latest checkpoints:"
                find "$OUTPUT_DIR" -name "checkpoint-*" -type d -printf '%T@ %p\n' 2>/dev/null | \
                sort -rn | head -3 | cut -d' ' -f2 | xargs -I {} basename {}
            fi
        fi
        echo ""

        # Disk usage
        section_header "Disk Usage"
        if [ -d "$OUTPUT_DIR" ]; then
            du -sh "$OUTPUT_DIR" 2>/dev/null || echo "Cannot calculate size"
        else
            echo "Output directory not found"
        fi
        echo ""

        # TensorBoard info
        section_header "TensorBoard"
        if pgrep -f "tensorboard.*$OUTPUT_DIR" > /dev/null; then
            echo -e "${GREEN}✓ TensorBoard is RUNNING${NC}"
            echo "View at: http://localhost:6006"
        else
            echo "TensorBoard is not running"
            echo "Start with: tensorboard --logdir $OUTPUT_DIR --port 6006"
        fi
        echo ""

        # Controls
        echo "========================================================================"
        echo "Press Ctrl+C to exit monitoring"
        echo "Refresh in 10 seconds..."
        echo "========================================================================"

        # Wait 10 seconds or until interrupted
        sleep 10
    done
}

# Start monitoring
monitor_loop
