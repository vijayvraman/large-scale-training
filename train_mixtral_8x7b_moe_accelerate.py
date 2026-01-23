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
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    MixtralForCausalLM,
    MixtralConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from tqdm.auto import tqdm
from supervised_routing import (
    SupervisedMoEWrapper,
    ExpertLabelEmbedding,
    compute_routing_supervision_loss,
    get_expert_utilization_stats,
)

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
        default=2,
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
        help="[DEPRECATED] Attempt to convert model FFN layers to MoE (experimental, not used with Mixtral)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    # Supervised Routing Arguments
    parser.add_argument(
        "--routing_loss_weight",
        type=float,
        default=0.1,
        help="Weight for supervised routing loss (beta in loss equation). Higher = stronger supervision. Range: 0.05-0.5"
    )
    parser.add_argument(
        "--disable_supervised_routing",
        action="store_true",
        help="Disable supervised routing and use standard MoE training without expert label guidance"
    )
    parser.add_argument(
        "--label_embedding_temperature",
        type=float,
        default=1.0,
        help="Temperature for expert label embedding softmax. Lower = harder category assignments. Range: 0.5-2.0"
    )
    return parser.parse_args()

# ---------------------------
# Helpers: load annotated data
# ---------------------------
# Expert Label Mapping
# ---------------------------
# Map expert label strings to integer IDs for supervised routing
EXPERT_LABEL_MAP = {
    "factual_lookup": 0,
    "numerical_reasoning": 1,
    "multi_hop_reasoning": 2,
    "commonsense_reasoning": 3,
}

def expert_label_to_id(label: str) -> int:
    """Convert expert label string to integer ID."""
    return EXPERT_LABEL_MAP.get(label, 0)  # Default to factual_lookup

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
        expert_label = record.get("expert_label", "factual_lookup")

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

        # Convert expert label to ID
        expert_label_id = expert_label_to_id(expert_label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "expert_label_ids": torch.tensor(expert_label_id, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "expert_label_ids": torch.stack([b["expert_label_ids"] for b in batch]),
    }


def get_dataloader_with_epoch_seed(dataset, batch_size, collate_fn, seed, epoch):
    """
    Create a DataLoader with deterministic shuffling based on epoch seed.
    This allows reproducible data ordering when resuming training.
    """
    # Create a generator with epoch-specific seed
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=generator,
    )

    return dataloader

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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,  # Use fast tokenizer for better performance
    )

    # Mixtral uses <s> for BOS and </s> for EOS
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load Mixtral-8x7B MoE model
    logger.info(f"Loading Mixtral MoE model from {args.model_id}")

    # Load model configuration with router logits enabled
    logger.info("Loading Mixtral config with router logits enabled")
    config = MixtralConfig.from_pretrained(
        args.model_id,
        output_router_logits=True,  # CRITICAL: Enable router logit output for supervised routing
        router_aux_loss_coef=0.01,  # Load balancing loss weight (alpha in loss equation)
        use_cache=False,  # Required for gradient checkpointing (set explicitly to avoid warnings)
    )
    logger.info(f"Mixtral config loaded: {config.num_local_experts} experts, router_jitter_noise={config.router_jitter_noise}")

    # Load Mixtral model with bfloat16
    try:
        model = MixtralForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability and memory efficiency
            device_map=None,  # Let Accelerate handle device placement
        )
        logger.info(f"Mixtral model loaded successfully with bfloat16")
    except Exception as e:
        logger.warning(f"Failed to load with bfloat16, trying float16: {e}")
        model = MixtralForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            torch_dtype=torch.float16,
            device_map=None,
        )
        logger.info(f"Mixtral model loaded with float16")

    # Report model size
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Model loaded. Total parameters: {total_params:.2f}B")
    logger.info(f"Active parameters per forward pass (2 of 8 experts): ~{total_params * 2 / 8:.2f}B")

    # Enable gradient checkpointing to reduce memory usage
    try:
        logger.info("Enabling gradient checkpointing for memory efficiency")
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled successfully")
    except (ValueError, AttributeError) as e:
        logger.warning(f"Gradient checkpointing not supported: {e}")
        logger.info("Continuing without gradient checkpointing")

    # Wrap model with supervised routing wrapper
    if not args.disable_supervised_routing:
        logger.info("=" * 60)
        logger.info("Wrapping model with supervised routing capability")
        logger.info(f"  - Routing loss weight: {args.routing_loss_weight}")
        logger.info(f"  - Label embedding temperature: {args.label_embedding_temperature}")
        logger.info(f"  - Dataset categories: {len(EXPERT_LABEL_MAP)} (factual, numerical, multi_hop, commonsense)")
        logger.info(f"  - Model experts: {config.num_local_experts}")
        logger.info("=" * 60)

        model = SupervisedMoEWrapper(
            moe_model=model,
            num_categories=4,  # factual, numerical, multi_hop, commonsense
            num_experts=config.num_local_experts,  # Mixtral has 8 experts
            routing_loss_weight=args.routing_loss_weight,
            label_embedding_temperature=args.label_embedding_temperature,
            router_aggregation="mean",  # Average router logits across all MoE layers
        )
        logger.info("Supervised routing wrapper initialized successfully")
    else:
        logger.info("Supervised routing disabled - using standard MoE training")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Calculate total training steps
    # We need to account for multi-GPU sharding, so create a temporary dataloader
    # to get the actual number of batches per device
    temp_dataloader = get_dataloader_with_epoch_seed(
        dataset=train_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=collate_fn,
        seed=args.seed,
        epoch=0,
    )
    temp_dataloader = accelerator.prepare(temp_dataloader)
    num_batches_per_epoch_per_device = len(temp_dataloader)

    # Use accelerator's actual gradient_accumulation_steps (may differ from args if using DeepSpeed config)
    actual_gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    if actual_gradient_accumulation_steps != args.gradient_accumulation_steps:
        logger.warning(
            f"Gradient accumulation steps mismatch! "
            f"args={args.gradient_accumulation_steps}, "
            f"accelerator (from DeepSpeed config)={actual_gradient_accumulation_steps}. "
            f"Using accelerator value for calculations."
        )

    num_update_steps_per_epoch = math.ceil(
        num_batches_per_epoch_per_device / actual_gradient_accumulation_steps
    )
    total_training_steps = args.epochs * num_update_steps_per_epoch
    del temp_dataloader  # Free memory

    # Initialize training state
    starting_epoch = 0
    global_step = 0
    steps_in_current_epoch = 0

    # Check if we're resuming from a checkpoint
    if args.resume_from_checkpoint:
        # Load checkpoint metadata to get the global_step
        metadata_path = os.path.join(args.resume_from_checkpoint, "training_state.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            global_step = metadata.get("global_step", 0)
            starting_epoch = metadata.get("epoch", 0)
            steps_in_current_epoch = global_step - (starting_epoch * num_update_steps_per_epoch)
            logger.info(f"Found checkpoint at global_step={global_step}, epoch={starting_epoch}")
        else:
            logger.warning(f"Checkpoint metadata not found at {metadata_path}, starting from scratch")

    # Calculate remaining training steps for learning rate scheduler
    remaining_steps = total_training_steps - global_step

    logger.info(f"Creating LR scheduler: total_steps={total_training_steps}, "
                f"completed_steps={global_step}, remaining_steps={remaining_steps}, "
                f"warmup_steps={args.warmup_steps}")

    # Learning rate scheduler - use cosine annealing for smoother decay
    # Cosine decay stays above zero longer and provides better convergence
    logger.info("Using cosine annealing schedule for learning rate decay")
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Prepare model, optimizer, and scheduler with accelerator
    # Note: We'll prepare the dataloader per epoch to allow epoch-based seeding
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    # Load checkpoint if resuming (load optimizer/scheduler states after preparing)
    if args.resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        loaded_global_step, loaded_epoch = load_checkpoint(
            checkpoint_dir=args.resume_from_checkpoint,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,  # Load scheduler state for proper LR resumption
        )
        # Verify the loaded values match what we read earlier
        if loaded_global_step != global_step:
            logger.warning(f"Global step mismatch: expected {global_step}, got {loaded_global_step}")
            global_step = loaded_global_step
            starting_epoch = loaded_epoch
            steps_in_current_epoch = global_step - (starting_epoch * num_update_steps_per_epoch)
        logger.info(f"Resuming from epoch {starting_epoch}, global_step {global_step}, steps_in_current_epoch {steps_in_current_epoch}")
        logger.info(f"Resumed learning rate: {lr_scheduler.get_last_lr()[0]:.2e}")

    logger.info("=" * 50)
    logger.info(f"Total examples: {len(train_dataset)}")
    logger.info(f"Examples per device (after sharding): {num_batches_per_epoch_per_device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Starting epoch: {starting_epoch}")
    logger.info(f"Starting global_step: {global_step}")
    logger.info(f"Batch size per device: {args.per_device_batch_size}")
    logger.info(f"Gradient accumulation steps: {actual_gradient_accumulation_steps}")
    logger.info(f"Number of devices: {accelerator.num_processes}")
    logger.info(f"Total batch size (with parallel & accumulation): {args.per_device_batch_size * accelerator.num_processes * actual_gradient_accumulation_steps}")
    logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Total optimization steps: {total_training_steps}")
    logger.info("=" * 50)

    # Initialize TensorBoard tracker
    accelerator.init_trackers(project_name="mpt7b_moe_training")
    logger.info(f"TensorBoard logging initialized. Logs will be written to: {args.output_dir}")

    # Training loop
    logger.info("Starting training...")
    total_loss = 0.0
    total_routing_loss = 0.0
    routing_loss_count = 0  # Track number of routing losses logged

    for epoch in range(starting_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        # Create dataloader with deterministic epoch-based seeding
        train_dataloader = get_dataloader_with_epoch_seed(
            dataset=train_dataset,
            batch_size=args.per_device_batch_size,
            collate_fn=collate_fn,
            seed=args.seed,
            epoch=epoch,
        )

        # Prepare dataloader with accelerator
        train_dataloader = accelerator.prepare(train_dataloader)

        # Calculate number of batches to skip if resuming mid-epoch
        batches_to_skip = 0
        if epoch == starting_epoch and steps_in_current_epoch > 0:
            # We need to skip batches we've already processed
            # Note: The dataloader is sharded across num_processes, so each process
            # only sees a fraction of the total batches
            total_batches_to_skip = steps_in_current_epoch * actual_gradient_accumulation_steps
            batches_to_skip = total_batches_to_skip // accelerator.num_processes
            logger.info(f"Resuming mid-epoch: skipping {batches_to_skip} batches per process "
                       f"({total_batches_to_skip} total batches across {accelerator.num_processes} processes)")

        # Calculate total batches and current position for progress tracking
        # Use total_training_steps to ensure batch progress aligns with step progress
        total_batches = total_training_steps * actual_gradient_accumulation_steps
        batches_completed_so_far = global_step * actual_gradient_accumulation_steps

        progress_bar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
            initial=batches_completed_so_far,
        )

        for step, batch in enumerate(train_dataloader):
            # Skip already-processed batches when resuming (without updating tqdm)
            if step < batches_to_skip:
                continue

            # Update progress bar only for actual training batches
            progress_bar.update(1)

            with accelerator.accumulate(model):
                # Forward pass (no need to move to device, accelerator handles it)
                outputs = model(**batch)
                loss = outputs.loss

                # Extract routing supervision loss if available (for monitoring)
                routing_loss = getattr(outputs, 'routing_supervision_loss', None)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer step
                optimizer.step()

                # Step scheduler after optimizer (only when actually stepping)
                if accelerator.sync_gradients:
                    lr_scheduler.step()

                optimizer.zero_grad()

            # Accumulate loss
            total_loss += loss.detach().item()
            epoch_loss += loss.detach().item()

            # Accumulate routing supervision loss if available
            if routing_loss is not None:
                total_routing_loss += routing_loss.detach().item()
                routing_loss_count += 1

            if accelerator.sync_gradients:
                global_step += 1

                # Check if we've completed training
                if global_step >= total_training_steps:
                    logger.info(f"Reached total_training_steps ({total_training_steps}), saving final checkpoint and stopping training.")
                    # Save final checkpoint before stopping
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir,
                        step=global_step,
                        epoch=epoch + 1,
                        args=args,
                    )
                    progress_bar.close()
                    break

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps

                    # Prepare metrics dictionary
                    metrics = {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }

                    # Add routing supervision loss if available
                    if routing_loss_count > 0:
                        avg_routing_loss = total_routing_loss / routing_loss_count
                        metrics["train/routing_supervision_loss"] = avg_routing_loss

                    # Log metrics to TensorBoard
                    accelerator.log(metrics, step=global_step)

                    # Console logging
                    log_str = (
                        f"Step {global_step}/{total_training_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    if routing_loss_count > 0:
                        log_str += f" | Routing Loss: {avg_routing_loss:.4f}"

                    logger.info(log_str)

                    # Reset accumulators
                    total_loss = 0.0
                    total_routing_loss = 0.0
                    routing_loss_count = 0

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        tokenizer=tokenizer,
                        output_dir=args.output_dir,
                        step=global_step,
                        epoch=None,  # Use step-based naming
                        current_epoch=epoch,  # But save current epoch in metadata
                        args=args,
                    )

                # Update progress bar
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                }
                if routing_loss is not None:
                    postfix["routing_loss"] = f"{routing_loss.item():.4f}"

                progress_bar.set_postfix(postfix)

        # Close progress bar
        progress_bar.close()

        # Check if we broke out early due to reaching total steps
        if global_step >= total_training_steps:
            logger.info(f"Training complete at {global_step} steps")
            break

        # Reset steps_in_current_epoch after completing the first epoch
        if epoch == starting_epoch:
            steps_in_current_epoch = 0

        # End of epoch
        avg_epoch_loss = epoch_loss / (len(train_dataloader) - batches_to_skip)

        # Log epoch metrics to TensorBoard
        accelerator.log({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch": epoch + 1,
        }, step=global_step)

        logger.info(f"Epoch {epoch + 1} completed | Average loss: {avg_epoch_loss:.4f}")

        # Save end-of-epoch checkpoint
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            step=global_step,
            epoch=epoch + 1,
            args=args,
        )

    logger.info("Training completed!")
    logger.info(f"Final model saved to {args.output_dir}")

    # Close TensorBoard tracker
    accelerator.end_training()
    logger.info("TensorBoard logging closed")


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    lr_scheduler,
    tokenizer,
    output_dir: str,
    step: int = None,
    epoch: int = None,
    current_epoch: int = None,
    args=None,
):
    """Save complete training checkpoint including model, optimizer, scheduler, and RNG states.

    Args:
        step: Global step number
        epoch: If provided, uses epoch-based naming (checkpoint-epoch-{epoch})
        current_epoch: The current epoch number to save in metadata (can differ from epoch param)
    """
    accelerator.wait_for_everyone()

    if epoch is not None:
        save_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        epoch_to_save = epoch  # For epoch checkpoints, use epoch value
    else:
        save_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
        epoch_to_save = current_epoch if current_epoch is not None else 0  # For step checkpoints, use current_epoch

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_dir}")

    # Save model using save_pretrained (only on main process)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved model and tokenizer to {save_dir}")

    # Use Accelerate's save_state for optimizer/scheduler (handles DeepSpeed ZeRO properly)
    accelerator_state_dir = os.path.join(save_dir, "accelerator_state")
    accelerator.save_state(accelerator_state_dir)

    if accelerator.is_main_process:
        logger.info(f"Saved optimizer/scheduler state to {accelerator_state_dir}")

        # Save RNG states
        rng_path = os.path.join(save_dir, "rng_state.pt")
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        accelerator.save(rng_state, rng_path)
        logger.info(f"Saved RNG states to {rng_path}")

        # Save training metadata
        metadata = {
            "global_step": step,
            "epoch": epoch_to_save,
            "completed_epoch": epoch if epoch is not None else -1,  # Only completed if this is an epoch checkpoint
        }
        if args is not None:
            metadata["args"] = vars(args)

        metadata_path = os.path.join(save_dir, "training_state.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved training metadata to {metadata_path}")

        logger.info(f"Checkpoint saved successfully to {save_dir}")

    accelerator.wait_for_everyone()


def load_checkpoint(
    checkpoint_dir: str,
    accelerator: Accelerator,
    model,
    optimizer,
    lr_scheduler,
):
    """Load complete training checkpoint including model, optimizer, scheduler, and RNG states."""
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Load training metadata
    metadata_path = os.path.join(checkpoint_dir, "training_state.json")
    if not os.path.exists(metadata_path):
        raise ValueError(f"Training state file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    global_step = metadata.get("global_step", 0)
    epoch = metadata.get("epoch", 0)

    logger.info(f"Resuming from global_step={global_step}, epoch={epoch}")

    # Load model weights
    logger.info("Loading model weights...")
    unwrapped_model = accelerator.unwrap_model(model)

    # Check if this is a sharded checkpoint or single file
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    single_safetensors = os.path.join(checkpoint_dir, "model.safetensors")
    single_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(index_file) or os.path.exists(single_safetensors) or os.path.exists(single_bin):
        # Use from_pretrained to handle both sharded and single-file checkpoints
        logger.info("Loading model weights using from_pretrained (handles sharded checkpoints)...")
        state_dict = unwrapped_model.__class__.from_pretrained(
            checkpoint_dir,
            state_dict=None,
            local_files_only=True,
        ).state_dict()
        unwrapped_model.load_state_dict(state_dict, strict=False)
        logger.info("Model weights loaded successfully")
    else:
        raise ValueError(f"No model file found in {checkpoint_dir}")

    # Load optimizer/scheduler state
    # First try new format (accelerator_state dir)
    accelerator_state_dir = os.path.join(checkpoint_dir, "accelerator_state")

    if lr_scheduler is None:
        # Skip loading scheduler state - we created a fresh one for resumption
        logger.info("Skipping scheduler state loading (using fresh scheduler for resumption)")

        # Load only optimizer state from accelerator_state
        if os.path.exists(accelerator_state_dir):
            try:
                # Load optimizer state manually from the accelerator_state directory
                optimizer_state_pattern = os.path.join(accelerator_state_dir, "pytorch_model", "*_optim_states.pt")
                import glob
                optimizer_files = glob.glob(optimizer_state_pattern)
                if optimizer_files:
                    logger.info(f"Found optimizer state files, loading via accelerator.load_state")
                    # Note: accelerator.load_state will load both optimizer and scheduler
                    # but since we're not using the old scheduler, the loaded scheduler state won't affect us
                    accelerator.load_state(accelerator_state_dir)
                    logger.info("Optimizer state loaded successfully from accelerator_state")
                else:
                    logger.warning("No optimizer state files found in accelerator_state")
            except Exception as e:
                logger.warning(f"Failed to load accelerator state: {e}")
                logger.warning("Continuing with fresh optimizer state.")
        else:
            # Fall back to old format (separate optimizer.pt file)
            logger.info("Accelerator state directory not found, trying legacy checkpoint format...")
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                try:
                    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
                    optimizer.load_state_dict(optimizer_state)
                    logger.info("Optimizer state loaded successfully (legacy format)")
                except (KeyError, RuntimeError) as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
                    logger.warning("Continuing with fresh optimizer state.")
            else:
                logger.warning(f"Optimizer state not found, starting with fresh optimizer")
    else:
        # Load both optimizer and scheduler state (normal resumption)
        if os.path.exists(accelerator_state_dir):
            try:
                accelerator.load_state(accelerator_state_dir)
                logger.info("Optimizer and scheduler state loaded successfully from accelerator_state")
            except Exception as e:
                logger.warning(f"Failed to load accelerator state: {e}")
                logger.warning("Continuing with fresh optimizer/scheduler state.")
        else:
            # Fall back to old format (separate optimizer.pt and scheduler.pt files)
            logger.info("Accelerator state directory not found, trying legacy checkpoint format...")

            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                try:
                    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
                    optimizer.load_state_dict(optimizer_state)
                    logger.info("Optimizer state loaded successfully (legacy format)")
                except (KeyError, RuntimeError) as e:
                    logger.warning(f"Failed to load optimizer state (likely due to DeepSpeed ZeRO sharding mismatch): {e}")
                    logger.warning("Continuing with fresh optimizer state. Momentum will be reset.")
            else:
                logger.warning(f"Optimizer state not found, starting with fresh optimizer")

            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            if os.path.exists(scheduler_path):
                try:
                    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
                    lr_scheduler.load_state_dict(scheduler_state)
                    logger.info("Scheduler state loaded successfully (legacy format)")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
            else:
                logger.warning(f"Scheduler state not found, starting with fresh scheduler")

    # Load RNG states
    rng_path = os.path.join(checkpoint_dir, "rng_state.pt")
    if os.path.exists(rng_path):
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        logger.info("RNG states restored successfully")
    else:
        logger.warning(f"RNG state not found at {rng_path}, random states not restored")

    logger.info(f"Checkpoint loaded successfully from {checkpoint_dir}")

    return global_step, epoch


if __name__ == "__main__":
    main()

