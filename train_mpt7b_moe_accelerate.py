"""
train_mpt7b_moe_accelerate.py
Train MPT-7B style model with DeepSpeed-MoE using HuggingFace Accelerate.

Run:
  accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py

Or with more options:
  accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py \
    --model_id mosaicml/mpt-7b \
    --data_file nq_annotated_moe.jsonl \
    --output_dir ./mpt7b_moe_finetune \
    --epochs 3 \
    --learning_rate 2e-5 \
    --max_seq_length 512

Notes:
- Ensure deepspeed is installed with MoE support: pip install deepspeed
- Adjust num_processes in accelerate_config.yaml to match your number of GPUs
- Adjust batch sizes in deepspeed_moe_config.json for your hardware
"""

import os
import math
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train MPT-7B-MoE with Accelerate")
    parser.add_argument(
        "--model_id",
        type=str,
        default="mosaicml/mpt-7b",
        help="HuggingFace model ID to use as base model"
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Specific model revision/commit to use (helps with compatibility)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="nq_annotated_moe.jsonl",
        help="Path to annotated JSONL dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mpt7b_moe_finetune",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Maximum sequence length for input"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Maximum length for target/answer"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU (micro batch)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples for quick testing"
    )
    parser.add_argument(
        "--convert_to_moe",
        action="store_true",
        help="Attempt to convert model FFN layers to MoE (experimental)"
    )
    return parser.parse_args()

# ---------------------------
# Helpers: load annotated data
# ---------------------------
def load_annotated_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load annotated JSONL dataset with question, answer, and expert_label fields."""
    records = []
    logger.info(f"Loading data from {path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {i+1}: {e}")
                continue

            # Expect fields: question, answer, expert_label
            q = obj.get("question", "").strip()
            a = obj.get("answer", "")

            # Normalize answer field: if it's dict/list follow your format
            if isinstance(a, dict):
                # Try common NQ structure
                a_text = a.get("text", "")
            elif isinstance(a, list):
                a_text = ", ".join([str(x) for x in a])
            else:
                a_text = str(a)

            if not q or not a_text:
                logger.warning(f"Skipping record {i+1} with empty question or answer")
                continue

            records.append({
                "question": q,
                "answer": a_text,
                "expert_label": obj.get("expert_label", "factual_lookup")
            })

    logger.info(f"Loaded {len(records)} records")
    return records

# ---------------------------
# Tokenization / Dataset
# ---------------------------
class QADataset(torch.utils.data.Dataset):
    """Dataset for Question-Answer pairs with causal LM training."""

    def __init__(self, records: List[Dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        question = record["question"]
        answer = record["answer"]

        # Format: "Question: {q}\nAnswer: {a}"
        prompt = f"Question: {question}\nAnswer:"
        full_text = f"{prompt} {answer}"

        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels: mask out prompt tokens (only compute loss on answer)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
        )
        prompt_len = len(prompt_encoding["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

# ---------------------------
# Replace FFN with DeepSpeed MoE layers (optional, experimental)
# ---------------------------
def convert_model_to_moe(model, num_experts: int = 4):
    """
    Experimental function to convert model FFN layers to DeepSpeed MoE.

    WARNING: This is experimental and depends heavily on:
    1. DeepSpeed version and MoE API
    2. Model architecture (MPT, GPT, etc.)
    3. Exact module names in the model

    For production use, consider:
    - Using a pre-trained MoE model
    - Manual architecture modification
    - DeepSpeed's built-in MoE conversion tools

    Args:
        model: HuggingFace model to convert
        num_experts: Number of experts for MoE layers

    Returns:
        model: Modified model with MoE layers (or original if conversion fails)
    """
    try:
        import deepspeed
        from deepspeed.moe.layer import MoE
    except ImportError as e:
        logger.error(f"DeepSpeed MoE not available: {e}")
        logger.error("Install with: pip install deepspeed")
        return model

    logger.info(f"Attempting to convert model to MoE with {num_experts} experts")
    logger.warning("This is experimental and may not work for all model architectures")

    replaced = 0

    # Try to find and replace FFN/MLP modules
    # This example targets MPT-style models with transformer.blocks[i].ffn or .mlp
    if hasattr(model, "transformer"):
        transformer = model.transformer

        # Check for blocks (MPT-style)
        if hasattr(transformer, "blocks"):
            for i, block in enumerate(transformer.blocks):
                if hasattr(block, "ffn"):
                    old_ffn = block.ffn
                    try:
                        # Get hidden dimension from first layer
                        if hasattr(old_ffn, "up_proj"):
                            hidden_size = old_ffn.up_proj.in_features
                            intermediate_size = old_ffn.up_proj.out_features
                        elif hasattr(old_ffn, "fc1"):
                            hidden_size = old_ffn.fc1.in_features
                            intermediate_size = old_ffn.fc1.out_features
                        else:
                            logger.warning(f"Could not determine FFN dimensions for block {i}")
                            continue

                        logger.info(f"Replacing FFN in block {i}: hidden={hidden_size}, intermediate={intermediate_size}")

                        # This is a simplified example - actual MoE layer creation
                        # depends on DeepSpeed version and requirements
                        # You may need to adjust parameters
                        moe_layer = MoE(
                            hidden_size=hidden_size,
                            expert=old_ffn,  # Use existing FFN as expert template
                            num_experts=num_experts,
                            k=1,  # Top-K routing
                        )
                        block.ffn = moe_layer
                        replaced += 1

                    except Exception as e:
                        logger.warning(f"Failed to replace FFN in block {i}: {e}")
                        continue

    if replaced > 0:
        logger.info(f"Successfully replaced {replaced} FFN modules with MoE layers")
    else:
        logger.warning("No FFN modules were replaced. Model structure may not match expected pattern.")
        logger.info("Proceeding with original model architecture.")

    return model

# ---------------------------
# Training loop
# ---------------------------
def main():
    # Parse arguments
    args = parse_args()

    # Initialize accelerator (will pick up DeepSpeed config from accelerate config)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # Set up logging on main process only
    if accelerator.is_main_process:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Log configuration
    logger.info("=" * 50)
    logger.info("Training Configuration")
    logger.info("=" * 50)
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            logger.info("Added new pad token: <|pad|>")

    # Load model
    logger.info(f"Loading model from {args.model_id}")

    # Patch HuggingFace cache to fix compatibility issues
    # This needs to happen BEFORE any model loading
    def patch_mpt_model_files():
        """Patch downloaded MPT model files to fix compatibility issues."""
        import glob
        import sys

        # Try multiple times as files may still be downloading
        for attempt in range(3):
            cache_dirs = glob.glob("/home/ubuntu/.cache/huggingface/modules/transformers_modules/mosaicml/mpt*/*/")

            patched_any = False
            for cache_dir in cache_dirs:
                # Patch modeling_mpt.py for LlamaDynamicNTKScalingRotaryEmbedding import
                modeling_file = os.path.join(cache_dir, "modeling_mpt.py")
                if os.path.exists(modeling_file):
                    with open(modeling_file, 'r') as f:
                        content = f.read()

                    if "HFDynamicNTKScalingRotaryEmbedding = None" not in content:
                        # Patch all three Llama imports
                        imports_to_patch = [
                            ("from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding as HFDynamicNTKScalingRotaryEmbedding",
                             """try:
    from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding as HFDynamicNTKScalingRotaryEmbedding
except ImportError:
    HFDynamicNTKScalingRotaryEmbedding = None"""),
                            ("from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding as HFLinearScalingRotaryEmbedding",
                             """try:
    from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding as HFLinearScalingRotaryEmbedding
except ImportError:
    HFLinearScalingRotaryEmbedding = None"""),
                            ("from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFRotaryEmbedding",
                             """try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFRotaryEmbedding
except ImportError:
    HFRotaryEmbedding = None""")
                        ]

                        for old_import, new_import in imports_to_patch:
                            content = content.replace(old_import, new_import)

                        with open(modeling_file, 'w') as f:
                            f.write(content)
                        logger.info(f"Patched {modeling_file}")
                        patched_any = True

                        # Clear any cached imports of this module
                        module_path_base = cache_dir.replace("/home/ubuntu/.cache/huggingface/modules/", "").replace("/", ".")
                        for key in list(sys.modules.keys()):
                            if module_path_base in key:
                                del sys.modules[key]
                                logger.info(f"Cleared cached module: {key}")

                # Create stub flash_attn_triton.py if missing
                flash_attn_file = os.path.join(cache_dir, "flash_attn_triton.py")
                if not os.path.exists(flash_attn_file):
                    with open(flash_attn_file, 'w') as f:
                        f.write('''"""Stub file for flash_attn_triton to satisfy import checks."""
def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("Flash attention with Triton is not available. Use attn_impl='torch'.")
''')
                    logger.info(f"Created stub {flash_attn_file}")
                    patched_any = True

            if patched_any or len(cache_dirs) > 0:
                break
            time.sleep(0.5)

    # Pre-emptively try to patch any existing cached files
    # DISABLED: transformers 4.47.0 should be compatible
    # patch_mpt_model_files()

    # First, load and modify the config to use torch attention instead of flash attention
    # This avoids triton_pre_mlir dependency issues
    logger.info("Loading config and setting attn_impl to 'torch'")
    try:
        config = AutoConfig.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Initial config load failed, patching model files again: {e}")
        time.sleep(1)  # Give downloads time to complete
        patch_mpt_model_files()
        config = AutoConfig.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
    # Set attention implementation to torch
    if hasattr(config, 'attn_config'):
        config.attn_config['attn_impl'] = 'torch'
    else:
        config.attn_config = {'attn_impl': 'torch'}

    logger.info(f"Attention config: {config.attn_config}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
        )
    except Exception as e:
        logger.warning(f"Failed to load with bfloat16, trying float16: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    # Resize token embeddings if we added new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Enable gradient checkpointing to reduce memory usage (if supported)
    try:
        logger.info("Attempting to enable gradient checkpointing for memory efficiency")
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled successfully")
    except (ValueError, AttributeError) as e:
        logger.warning(f"Gradient checkpointing not supported for this model: {e}")
        logger.info("Continuing without gradient checkpointing")

    # Optionally convert model to MoE (experimental)
    if args.convert_to_moe:
        logger.info("Converting model to MoE (experimental)")
        try:
            # Read num_experts from deepspeed config if available
            deepspeed_config_path = "deepspeed_moe_config.json"
            num_experts = 4
            if os.path.exists(deepspeed_config_path):
                with open(deepspeed_config_path, "r") as f:
                    ds_config = json.load(f)
                    num_experts = ds_config.get("moe", {}).get("num_experts", 4)
                    logger.info(f"Using num_experts={num_experts} from DeepSpeed config")

            model = convert_model_to_moe(model, num_experts=num_experts)
        except Exception as e:
            logger.error(f"MoE conversion failed: {e}")
            logger.info("Continuing with original model architecture")

    # Load dataset
    logger.info(f"Loading dataset from {args.data_file}")
    records = load_annotated_jsonl(args.data_file, max_samples=args.max_samples)

    if len(records) == 0:
        logger.error("No valid records found in dataset!")
        return

    # Create dataset
    train_dataset = QADataset(
        records=records,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )
    logger.info(f"Created dataset with {len(train_dataset)} examples")

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenizer
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_training_steps = args.epochs * num_update_steps_per_epoch

    # Learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    logger.info("=" * 50)
    logger.info(f"Total examples: {len(train_dataset)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size per device: {args.per_device_batch_size}")
    logger.info(f"Total batch size (with parallel & accumulation): {args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_training_steps}")
    logger.info("=" * 50)

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    total_loss = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Forward pass (no need to move to device, accelerator handles it)
                outputs = model(**batch)
                loss = outputs.loss

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Accumulate loss
            total_loss += loss.detach().item()
            epoch_loss += loss.detach().item()

            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    logger.info(
                        f"Step {global_step}/{total_training_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    total_loss = 0.0

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir,
                        step=global_step,
                    )

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} completed | Average loss: {avg_epoch_loss:.4f}")

        # Save end-of-epoch checkpoint
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            step=global_step,
            epoch=epoch + 1,
        )

    logger.info("Training completed!")
    logger.info(f"Final model saved to {args.output_dir}")


def save_checkpoint(
    accelerator: Accelerator,
    model,
    tokenizer,
    output_dir: str,
    step: int = None,
    epoch: int = None,
):
    """Save model checkpoint."""
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if epoch is not None:
            save_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        else:
            save_dir = os.path.join(output_dir, f"checkpoint-step-{step}")

        os.makedirs(save_dir, exist_ok=True)

        # Unwrap model from DDP/FSDP wrappers
        unwrapped_model = accelerator.unwrap_model(model)

        # Save model
        logger.info(f"Saving checkpoint to {save_dir}")
        unwrapped_model.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        # Save tokenizer
        tokenizer.save_pretrained(save_dir)

        logger.info(f"Checkpoint saved successfully to {save_dir}")


if __name__ == "__main__":
    main()

