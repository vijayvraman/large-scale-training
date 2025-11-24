# MPT-7B-MoE Training with Accelerate

Train MPT-7B Mixture of Experts (MoE) models using HuggingFace Transformers, Accelerate, and DeepSpeed on multi-GPU systems.

## Features

- Multi-GPU distributed training using HuggingFace Accelerate
- DeepSpeed integration with MoE support
- ZeRO-2 optimization for memory efficiency
- Mixed precision training (FP16/BF16)
- Question-Answer dataset format support
- Automatic checkpointing and resume
- Gradient accumulation and clipping
- Comprehensive logging

## Requirements

- Python 3.8+
- CUDA-capable GPUs (tested with 4+ GPUs)
- 40GB+ GPU memory recommended per GPU for MPT-7B

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

The training script expects a JSONL file where each line contains:

```json
{"question": "Your question here", "answer": "Your answer here", "expert_label": "factual_lookup"}
```

Example (`nq_annotated_moe.jsonl`):
```json
{"question": "where did they film hot tub time machine", "answer": ["Fernie Alpine Resort"], "expert_label": "factual_lookup"}
{"question": "who has the right of way in international waters", "answer": ["Neither vessel"], "expert_label": "factual_lookup"}
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

Key settings to adjust:

```json
{
  "train_batch_size": 32,  // Total batch size across all GPUs
  "train_micro_batch_size_per_gpu": 1,  // Batch per GPU
  "gradient_accumulation_steps": 16,  // Accumulation steps
  "moe": {
    "num_experts": 4,  // Number of MoE experts
    "top_k": 1  // Top-k routing
  }
}
```

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

```bash
tensorboard --logdir ./mpt7b_moe_finetune
```

### GPU Monitoring

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Checkpoints

Checkpoints are saved to `{output_dir}/checkpoint-{step|epoch}-{N}/`:

```
mpt7b_moe_finetune/
├── checkpoint-step-500/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── checkpoint-epoch-1/
│   └── ...
```

### Loading Checkpoints

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./mpt7b_moe_finetune/checkpoint-epoch-1")
tokenizer = AutoTokenizer.from_pretrained("./mpt7b_moe_finetune/checkpoint-epoch-1")
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `per_device_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Reduce `max_seq_length`
4. Enable ZeRO-3 in DeepSpeed config
5. Try smaller model variants

### Slow Training

1. Increase `per_device_batch_size` if memory allows
2. Reduce `logging_steps` and `save_steps`
3. Use `num_workers > 0` in DataLoader (may need adjustment)
4. Ensure GPUs are properly utilized: check `nvidia-smi`

### Multi-GPU Issues

1. Verify all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
2. Check GPU count matches `num_processes` in accelerate config
3. Try: `accelerate config` to reconfigure
4. Use `NCCL_DEBUG=INFO` for detailed distributed training logs

### DeepSpeed/MoE Issues

1. Ensure DeepSpeed is installed: `pip install deepspeed`
2. Check CUDA compatibility: `python -c "import deepspeed; print(deepspeed.version)"`
3. The `--convert_to_moe` flag is experimental; for production, use pre-trained MoE models
4. MoE layer conversion depends on model architecture - may need manual adjustment

## Performance Tips

1. **Batch Size**: Effective batch size = `per_device_batch_size × num_gpus × gradient_accumulation_steps`
2. **Mixed Precision**: BF16 (bfloat16) is generally more stable than FP16
3. **Gradient Checkpointing**: Add to model config if OOM (trades compute for memory)
4. **Data Loading**: Keep `num_workers=0` initially to avoid tokenizer pickling issues
5. **Learning Rate**: For large batch sizes, scale LR proportionally

## File Structure

```
.
├── train_mpt7b_moe_accelerate.py  # Main training script
├── accelerate_config.yaml         # Accelerate configuration
├── deepspeed_moe_config.json      # DeepSpeed MoE configuration
├── nq_annotated_moe.jsonl         # Training dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

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

## Support

For issues:
- HuggingFace Accelerate: https://github.com/huggingface/accelerate
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- MPT Models: https://huggingface.co/mosaicml
