# MPT-7B-MoE Training with Accelerate

Train MPT-7B Mixture of Experts (MoE) models using HuggingFace Transformers, Accelerate, and DeepSpeed on multi-GPU systems.

## Overview

- **Model**: MPT-7B (mosaicml/mpt-7b)
- **Dataset**: Natural Questions (NQ Open) - 87,925 Q&A pairs
- **Hardware**: 2x NVIDIA H100 80GB GPUs
- **Training Framework**: DeepSpeed ZeRO Stage 2 with HuggingFace Accelerate
- **Expert Configuration**: 4 experts (factual_lookup, numerical_reasoning, multi_hop_reasoning, commonsense_reasoning)
- **Training Time**: ~2-2.5 hours per epoch (optimized)

## Features

- Multi-GPU distributed training using HuggingFace Accelerate
- DeepSpeed ZeRO Stage 2 optimization for optimal speed/memory balance
- Mixed precision training (BF16)
- Natural Questions dataset with expert routing annotations
- **Mid-epoch resumable training** with complete state preservation
- Deterministic data loading for reproducible training
- Automatic checkpointing with optimizer and scheduler state
- Gradient accumulation and clipping
- Comprehensive logging and monitoring
- Optimized for H100 GPUs

## Requirements

- Python 3.8+
- CUDA-capable GPUs (optimized for 2x H100 80GB)
- DeepSpeed, Transformers, Accelerate

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
```

## Dataset Format

We'll focus on the 4 reasoning-style experts:
- **Causal / Explanatory**: why, how, explain, cause
- **Comparative / Superlative**: largest, taller, vs, more than
- **Multi-hop**: multiple entities, conjunctions (and, or, between)
- **Direct Lookup**: everything else

The training script expects a JSONL file where each line contains:

```json
{"question": "Your question here", "answer": "Your answer here", "expert_label": "factual_lookup"}
```

Example (`nq_annotated_moe.jsonl`):
```json
{"question": "where did they film hot tub time machine", "answer": ["Fernie Alpine Resort"], "expert_label": "factual_lookup"}
{"question": "how many episodes in season 4 of the flash", "answer": ["23"], "expert_label": "numerical_reasoning"}
{"question": "who has won more grammy awards kelly or carrie", "answer": ["Carrie Underwood"], "expert_label": "multi_hop_reasoning"}
{"question": "why do we have daylight saving time in the us", "answer": ["to save energy"], "expert_label": "commonsense_reasoning"}
```

The `answer` field can be:
- A string: `"answer text"`
- A list: `["answer1", "answer2"]` (will be joined with commas)
- A dict: `{"text": "answer text"}` (common NQ format)

## Configuration

### 1. Accelerate Config (`accelerate_config.yaml`)

Adjust `num_processes` to match your number of GPUs:

```yaml
num_processes: 4  # Change to your GPU count
```

### 2. DeepSpeed Config (`deepspeed_moe_config.json`)

**Current Optimized Configuration (ZeRO Stage 2):**

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "bf16": {"enabled": true},
  "moe": {
    "enabled": true,
    "num_experts": 4,
    "top_k": 1
  }
}
```

**Key Configuration Details:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **ZeRO Stage** | 2 | Shards optimizer + gradients across GPUs (no CPU offload) |
| **Micro batch per GPU** | 4 | Actual batch size loaded per GPU |
| **Gradient accumulation** | 4 | Accumulate over 4 steps before optimizer update |
| **Effective batch size** | 32 | 2 GPUs × 4 micro batch × 4 accumulation |
| **Sequence length** | 256 | Maximum token length (set via --max_seq_length) |
| **Precision** | bfloat16 | Better numerical stability than fp16 |

## Training

### Basic Training

```bash
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py
```

### Training with Custom Options

```bash
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
    --model_id mosaicml/mpt-7b \
    --data_file nq_annotated_moe.jsonl \
    --output_dir ./mpt7b_moe_checkpoints \
    --epochs 3 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --save_steps 500
```

### Resuming Training from Checkpoint

```bash
# Resume from any checkpoint (mid-epoch or end-of-epoch)
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
    --resume_from_checkpoint ./mpt7b_moe_checkpoints/checkpoint-step-500 \
    --model_id mosaicml/mpt-7b \
    --data_file nq_annotated_moe.jsonl \
    --output_dir ./mpt7b_moe_checkpoints
```

**Note**: The training will automatically:
- Restore model, optimizer, and scheduler states
- Skip already-processed batches if resuming mid-epoch
- Continue with the same learning rate schedule
- Maintain reproducible data ordering

### All Available Arguments

```bash
# Model settings
--model_id                     HuggingFace model ID (default: mosaicml/mpt-7b)
--convert_to_moe               Attempt to convert model to MoE (experimental)

# Data settings
--data_file                    Path to JSONL dataset (default: nq_annotated_moe.jsonl)
--max_samples                  Limit samples for testing (default: None)
--max_seq_length              Maximum sequence length (default: 512)
--max_target_length           Maximum target length (default: 128)

# Training settings
--epochs                       Number of epochs (default: 1)
--learning_rate               Learning rate (default: 2e-5)
--per_device_batch_size       Batch size per GPU (default: 1)
--gradient_accumulation_steps Gradient accumulation (default: 8)
--warmup_steps                Warmup steps (default: 100)
--max_grad_norm               Max gradient norm (default: 1.0)

# Logging and checkpointing
--output_dir                  Output directory (default: ./mpt7b_moe_finetune)
--logging_steps               Log every N steps (default: 10)
--save_steps                  Save every N steps (default: 500)
--resume_from_checkpoint      Path to checkpoint directory to resume from (default: None)

# Reproducibility
--seed                        Random seed (default: 42)
```

### Quick Test Run

Test with limited samples:

```bash
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
    --max_samples 100 \
    --epochs 1 \
    --save_steps 50
```

## Monitoring

### TensorBoard

The training script automatically logs metrics to TensorBoard for real-time monitoring and analysis.

#### Logged Metrics

The following metrics are tracked during training:

**Step-level metrics** (logged every `--logging_steps`, default: 10):
- `train/loss`: Average training loss over the logging interval
- `train/learning_rate`: Current learning rate from the scheduler
- `train/epoch`: Current epoch number
- `train/global_step`: Global training step count

**Epoch-level metrics** (logged at the end of each epoch):
- `train/epoch_loss`: Average loss across the entire epoch
- `train/epoch`: Completed epoch number

#### Viewing Logs

TensorBoard logs are saved to `{output_dir}/mpt7b_moe_training/`. To launch TensorBoard:

```bash
# Default output directory
tensorboard --logdir ./mpt7b_moe_finetune

# Custom output directory
tensorboard --logdir ./mpt7b_moe_checkpoints

# Specify port
tensorboard --logdir ./mpt7b_moe_finetune --port 6006
```

Then open your browser to `http://localhost:6006` to view the training metrics.

#### What to Look For

- **Loss curve**: Should gradually decrease over time
- **Learning rate**: Should follow warmup → linear decay schedule
- **Loss spikes**: Occasional spikes are normal, but persistent increases indicate issues
- **Plateau**: If loss stops decreasing, consider adjusting learning rate or checking data quality

### GPU Monitoring

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Checkpoints

### Checkpoint Structure

Checkpoints now include complete training state for mid-epoch resumption:

```
mpt7b_moe_finetune/
├── checkpoint-step-500/
│   ├── config.json              # Model configuration
│   ├── model.safetensors        # Model weights
│   ├── optimizer.pt             # Optimizer state (Adam moments, etc.)
│   ├── scheduler.pt             # Learning rate scheduler state
│   ├── rng_state.pt             # Random number generator states
│   ├── training_state.json      # Training metadata (step, epoch, args)
│   └── tokenizer files...
├── checkpoint-epoch-1/
│   └── ...
```

### Resumable Training

Training can now be resumed from any checkpoint, including mid-epoch checkpoints:

```bash
# Resume from a specific checkpoint
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
  --resume_from_checkpoint ./mpt7b_moe_finetune/checkpoint-step-500 \
  --model_id mosaicml/mpt-7b \
  --data_file nq_annotated_moe.jsonl \
  --output_dir ./mpt7b_moe_finetune
```

**Key Features:**
- **Mid-Epoch Resumption**: Resume training from any saved step, not just epoch boundaries
- **Deterministic Shuffling**: Uses epoch-based seeding to ensure reproducible data ordering
- **Complete State**: Restores optimizer state, learning rate scheduler, RNG states, and training progress
- **Automatic Skip**: Automatically skips already-processed batches when resuming mid-epoch

**What Gets Restored:**
- Model weights
- Optimizer state (momentum, adaptive learning rates)
- Learning rate scheduler state (warmup progress, step count)
- Training counters (global step, epoch)
- Random states (PyTorch, NumPy, Python) for reproducibility
- Data loading position within the epoch

### Loading Checkpoints for Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./mpt7b_moe_finetune/checkpoint-epoch-1")
tokenizer = AutoTokenizer.from_pretrained("./mpt7b_moe_finetune/checkpoint-epoch-1")
```

## Mid-Epoch Resumable Training

### Overview

The training script now supports complete mid-epoch resumption, allowing you to stop and restart training at any point without losing progress. This is crucial for:
- Long-running training jobs that may be interrupted
- Spot instance/preemptible VM usage for cost savings
- Handling hardware failures or maintenance windows
- Debugging and iterative development

### Implementation Details

#### 1. **Enhanced Checkpoint Saving** (`train_mpt7b_moe_accelerate.py:728-799`)

Every checkpoint now includes complete training state:

| Component | File | Description |
|-----------|------|-------------|
| Model weights | `model.safetensors` | Complete model parameters |
| Optimizer state | `optimizer.pt` | Adam momentum, adaptive learning rates |
| Scheduler state | `scheduler.pt` | Warmup progress, current step |
| RNG states | `rng_state.pt` | Python, NumPy, PyTorch, CUDA random states |
| Training metadata | `training_state.json` | Global step, epoch, training arguments |
| Tokenizer | `tokenizer.json`, etc. | Tokenizer configuration and vocabulary |

#### 2. **Checkpoint Loading** (`train_mpt7b_moe_accelerate.py:802-882`)

The `load_checkpoint` function restores all saved state:
- Automatically detects model file format (safetensors or bin)
- Loads optimizer and scheduler states onto correct devices
- Restores all random number generator states for reproducibility
- Returns global step and epoch information for resume logic

#### 3. **Deterministic Data Loading** (`train_mpt7b_moe_accelerate.py:279-297`)

Uses epoch-based seeding to ensure reproducible data ordering:
```python
def get_dataloader_with_epoch_seed(dataset, batch_size, collate_fn, seed, epoch):
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(dataset, shuffle=True, generator=generator, ...)
```

This ensures that:
- Each epoch has a deterministic shuffle order
- Resuming from epoch N will see the same data order as the original run
- Different epochs have different shuffle orders (seed + epoch)

#### 4. **Smart Batch Skipping** (`train_mpt7b_moe_accelerate.py:680-696`)

When resuming mid-epoch:
```python
batches_to_skip = steps_in_current_epoch * gradient_accumulation_steps
for step, batch in enumerate(dataloader):
    if step < batches_to_skip:
        continue  # Fast iteration, no computation
    # ... training logic
```

- Calculates exact number of batches to skip
- Uses fast iteration (no forward/backward passes)
- Minimal overhead even when skipping thousands of batches

### Usage Examples

#### Starting Fresh Training

```bash
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
  --model_id mosaicml/mpt-7b \
  --data_file nq_annotated_moe.jsonl \
  --output_dir ./mpt7b_moe_finetune \
  --epochs 3 \
  --learning_rate 2e-5 \
  --save_steps 500
```

This will save checkpoints:
- Every 500 steps: `checkpoint-step-500`, `checkpoint-step-1000`, etc.
- After each epoch: `checkpoint-epoch-1`, `checkpoint-epoch-2`, etc.

#### Resuming from Mid-Epoch Checkpoint

```bash
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
  --resume_from_checkpoint ./mpt7b_moe_finetune/checkpoint-step-500 \
  --model_id mosaicml/mpt-7b \
  --data_file nq_annotated_moe.jsonl \
  --output_dir ./mpt7b_moe_finetune
```

**Note**: When resuming, you still need to specify `--model_id` and `--data_file` for initialization, but the actual model weights and training state come from the checkpoint.

#### Resuming from End-of-Epoch Checkpoint

```bash
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
  --resume_from_checkpoint ./mpt7b_moe_finetune/checkpoint-epoch-1 \
  --model_id mosaicml/mpt-7b \
  --data_file nq_annotated_moe.jsonl \
  --output_dir ./mpt7b_moe_finetune
```

### What Happens During Resume

1. **Checkpoint Detection**: Script detects `--resume_from_checkpoint` argument
2. **State Restoration**: Loads all saved state (model, optimizer, scheduler, RNG states)
3. **Position Calculation**: Determines starting epoch and steps within current epoch
4. **Dataloader Recreation**: Creates dataloader with same epoch seed for reproducible shuffling
5. **Batch Skipping**: Fast-forwards through already-processed batches
6. **Seamless Continuation**: Training continues exactly where it left off

### Example Output When Resuming

```
Loading checkpoint from ./mpt7b_moe_finetune/checkpoint-step-500
Resuming from global_step=500, epoch=0
Model weights loaded successfully
Optimizer state loaded successfully
Scheduler state loaded successfully
RNG states restored successfully
Checkpoint loaded successfully

Starting epoch: 0
Starting global_step: 500
Resuming mid-epoch: skipping 4000 batches

Step 510/10000 | Loss: 2.1234 | LR: 1.98e-05
...
```

### Key Features

✅ **True Mid-Epoch Resumption**: Resume from any checkpoint, not just epoch boundaries
✅ **Deterministic & Reproducible**: Same data ordering and random states ensure identical results
✅ **Efficient Batch Skipping**: Fast iteration through processed batches (no computation)
✅ **Complete State Preservation**: Optimizer momentum, learning rate schedule, everything restored
✅ **Distributed Training Compatible**: Works seamlessly with Accelerate and DeepSpeed
✅ **Automatic Handling**: No manual configuration needed, just pass `--resume_from_checkpoint`

### Limitations and Considerations

1. **Checkpoint Size**: Checkpoints are larger (~2x model size) due to optimizer state
2. **Data File Requirement**: Must use the same dataset file when resuming
3. **Seed Dependency**: Changing `--seed` will result in different data ordering
4. **Gradient Accumulation**: Batch skipping accounts for gradient accumulation steps
5. **Multi-GPU Consistency**: Ensure same number of GPUs when resuming for best results

### Troubleshooting

**Issue**: "Training state file not found"
**Solution**: Checkpoint might be from old version. Use a newly saved checkpoint.

**Issue**: Loss jumps after resuming
**Solution**: Ensure you're using the same dataset and seed. Check that RNG states loaded successfully.

**Issue**: Slow resume when skipping many batches
**Solution**: This is expected. Skipping is fast (no computation) but iterating through DataLoader takes some time. Consider using more frequent checkpoints if this is a concern.

## Configuration Optimization Journey

### Evolution of Configuration

#### Initial Setup (ZeRO Stage 3 + CPU Offloading)
**Problem**: Very slow training (~7.5 hours per epoch)

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

- **GPU Usage**: Only 10 GB (12% utilization)
- **Speed**: ~1.57 it/s
- **Bottleneck**: CPU ↔ GPU transfers for parameters
- **Total iterations**: 43,963 per epoch

#### Why ZeRO Stage 2 Failed Initially
**OOM Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.38 GiB`

```
GPU has 79.19 GiB capacity, 9.03 GiB free
Process has 70.15 GiB in use (62.03 GiB by PyTorch)
```

**Root Cause**: Used `max_seq_length=512` which caused:
- 4x more activation memory (attention is O(seq_len²))
- Memory breakdown: 62 GB (model+optimizer+gradients) + 12 GB (activations) = 74 GB
- Exceeded 79 GB available on H100

#### Final Optimized Configuration (Current)
**Solution**: ZeRO Stage 2 with `max_seq_length=256` and `micro_batch_size=4`

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"}
  }
}
```

- **GPU Usage**: ~40-50 GB (50-60% utilization - safe margin)
- **Speed**: ~2-3 it/s (1.5-2x faster than Stage 3)
- **Training Time**: ~2-2.5 hours per epoch (3x faster)
- **Total iterations**: ~10,990 optimizer steps per epoch

### Memory Usage Breakdown

| Component | Size (BF16) | ZeRO-2 (per GPU) | ZeRO-3 w/ CPU offload |
|-----------|-------------|------------------|-----------------------|
| **Model Parameters** | ~14 GB | 7 GB (sharded) | ~0 GB (on CPU) |
| **Gradients** | ~14 GB | 7 GB (sharded) | 7 GB (sharded) |
| **Optimizer States** | ~28 GB | 14 GB (sharded) | ~0 GB (on CPU) |
| **Activations (batch=4, seq=256)** | ~10-15 GB | 10-15 GB | 10-15 GB |
| **TOTAL** | - | **~38-43 GB** | **~17-22 GB** |

## Understanding Training Metrics

### Progress Bar Explanation

```
Epoch 1/1: 5% | 2239/43963 [40:21<7:22:22, 1.57it/s, loss=0.0325, lr=1.97e-05]
11/25/2025 04:41:02 - INFO - Step 140/10991 | Loss: 0.7643 | LR: 1.97e-05
```

**Two different step counts:**
- **43,963**: Total dataloader iterations (batches processed)
  - Calculated as: 87,925 samples ÷ 2 (micro_batch × num_gpus) = 43,963
- **10,991**: Total optimizer steps (weight updates)
  - Calculated as: 43,963 ÷ 4 (gradient_accumulation_steps) = 10,991

**Loss values:**
- `loss=0.0325`: Instantaneous loss for current batch (can be volatile)
- `Loss: 0.7643`: Averaged loss over last 10 steps (more reliable metric)

### Batch Size Calculation

```
Effective Batch Size = num_gpus × micro_batch_size × gradient_accumulation_steps
                     = 2 × 4 × 4
                     = 32 samples per optimizer update
```

## Key Lessons Learned

### 1. Gradient Accumulation Does NOT Increase Memory

Gradient accumulation can be increased freely without memory penalty:
- Each micro-batch is processed independently
- Gradients are accumulated **in-place** (added to existing gradient tensors)
- Previous batch data is **freed from memory** before loading next batch
- Only ONE micro-batch is in GPU memory at any time

**Example**: `gradient_accumulation_steps: 4` vs `16` uses **same memory**

### 2. Sequence Length Has Quadratic Memory Impact

Memory scales with O(seq_len²) due to self-attention mechanism:
- **256 tokens**: 256² = 65,536 attention elements per head
- **512 tokens**: 512² = 262,144 attention elements per head
- **Result**: 4x more memory for activations!

### 3. ZeRO Stage Selection Guide

| Stage | GPU Memory | Speed | Use Case |
|-------|------------|-------|----------|
| **Stage 1** | High (60-70 GB) | Fastest | Maximum memory available |
| **Stage 2** | Medium (40-50 GB) | **Fast** | **Balanced (our choice)** |
| **Stage 3 (no offload)** | Low (30-40 GB) | Medium | Medium memory constraints |
| **Stage 3 + CPU offload** | Very Low (10-20 GB) | Slow | Extreme memory constraints |

**Rule of thumb**: Use the lowest ZeRO stage that fits in your GPU memory for best performance.

### 4. CPU Offloading Trade-off

CPU offloading saves GPU memory but at a significant cost:
- **Overhead**: PCIe bandwidth bottleneck (CPU ↔ GPU transfers)
- **Speed Impact**: Can make training 2-3x slower
- **When to use**: Only when absolutely necessary (GPU memory < 40 GB for 7B models)

For our H100s with 80 GB each, CPU offloading is **wasteful** - we have plenty of GPU memory!

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. **Reduce micro batch size**: `train_micro_batch_size_per_gpu: 4 → 2` or `1`
2. **Increase gradient accumulation**: `gradient_accumulation_steps: 4 → 8` (keeps same effective batch)
3. **Reduce sequence length**: `--max_seq_length 256 → 128`
4. **Enable CPU offloading** (last resort):
   ```json
   "offload_optimizer": {"device": "cpu", "pin_memory": true}
   ```
5. **Switch to ZeRO Stage 3** with parameter offloading

**Memory fragmentation fix** (if you see "reserved but unallocated" warnings):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Slow Training

If training is too slow:

1. **Check GPU memory usage**: Should be 50-70% utilized
   ```bash
   watch -n 2 nvidia-smi
   ```
2. **Increase micro batch size** if memory allows (more GPU usage = faster)
3. **Disable CPU offloading** if enabled
4. **Use ZeRO Stage 2** instead of Stage 3 (faster communication)
5. **Reduce logging frequency**: `--logging_steps 10 → 50`

### Multi-GPU Issues

1. Verify all GPUs are visible: `nvidia-smi`
2. Check GPU count matches `num_processes` in `accelerate_config.yaml`
3. Reconfigure if needed: `accelerate config`
4. Debug with: `NCCL_DEBUG=INFO accelerate launch ...`

## Performance Tips

1. **Maximize GPU Utilization**: Increase `train_micro_batch_size_per_gpu` until you use 60-80% of GPU memory
2. **Minimize Gradient Accumulation**: Lower values = faster training (use it only when memory-constrained)
3. **Sequence Length vs Batch Size**: Shorter sequences allow larger batch sizes
4. **Mixed Precision**: BF16 is more stable than FP16 for large models
5. **Gradient Checkpointing**: Already enabled in script - trades compute for memory
6. **Effective Batch Size Formula**:
   ```
   effective_batch = micro_batch_per_gpu × num_gpus × grad_accumulation
   ```

## File Structure

```
.
├── train_mpt7b_moe_accelerate.py  # Main training script
├── prepare_dataset.py              # Dataset annotation with expert labels
├── accelerate_config.yaml         # Accelerate distributed config (2 GPUs)
├── deepspeed_moe_config.json      # DeepSpeed ZeRO Stage 2 config
├── nq_annotated_moe.jsonl         # Annotated NQ dataset (87,925 samples)
├── mpt7b_moe_finetune/            # Output directory (checkpoints)
└── README.md                      # This file (comprehensive guide)
```

## Dataset Preparation

The Natural Questions dataset is annotated with expert labels using heuristic-based classification:

```bash
python prepare_dataset.py
```

This creates `nq_annotated_moe.jsonl` with 4 expert types:
- **factual_lookup** (who/what/when/where questions)
- **numerical_reasoning** (how many/much, distances, percentages)
- **multi_hop_reasoning** (relational/comparison questions)
- **commonsense_reasoning** (why/reason/cause questions)

The expert labels are used for MoE routing during training.

## Citation

If you use MPT models, please cite:

```bibtex
@software{mpt-7b,
  author = {MosaicML},
  title = {MPT-7B},
  year = {2023},
  url = {https://huggingface.co/mosaicml/mpt-7b}
}
```

## License

This training code is provided as-is. Please refer to the respective licenses of:
- HuggingFace Transformers
- DeepSpeed
- Accelerate
- MPT models (Apache 2.0)

## Quick Reference

### Current Optimized Settings
```bash
# Hardware
2x NVIDIA H100 80GB GPUs

# Configuration
ZeRO Stage: 2 (no CPU offload)
Micro batch per GPU: 4
Gradient accumulation: 4
Effective batch size: 32
Sequence length: 256
Precision: bfloat16

# Performance
GPU memory usage: ~40-50 GB per GPU
Training speed: ~2-3 it/s
Training time: ~2-2.5 hours per epoch
Total optimizer steps: ~10,991 per epoch
```

### Common Commands

```bash
# Start training
accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py

# Monitor GPUs
watch -n 2 nvidia-smi

# Check training logs
tail -f nohup.out | grep "Step.*Loss"

# Kill training
pkill -f train_mpt7b_moe_accelerate.py
```

### Configuration at a Glance

| What | Current Value | Where to Change |
|------|---------------|-----------------|
| ZeRO Stage | 2 | `deepspeed_moe_config.json` → `zero_optimization.stage` |
| Micro batch size | 4 | `deepspeed_moe_config.json` → `train_micro_batch_size_per_gpu` |
| Gradient accumulation | 4 | `deepspeed_moe_config.json` → `gradient_accumulation_steps` |
| Sequence length | 256 | Command line: `--max_seq_length 256` |
| Number of GPUs | 2 | `accelerate_config.yaml` → `num_processes` |
| Learning rate | 2e-5 | Command line: `--learning_rate 2e-5` |

## Support

For issues:
- HuggingFace Accelerate: https://github.com/huggingface/accelerate
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- MPT Models: https://huggingface.co/mosaicml

---

**Last Updated**: 2025-11-25
**Hardware**: 2x H100 80GB
**Status**: Optimized for ZeRO Stage 2 with 4x micro batch, ~2-2.5 hour training per epoch
