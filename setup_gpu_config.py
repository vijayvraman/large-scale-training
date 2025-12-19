#!/usr/bin/env python3
"""
setup_gpu_config.py
Automatically detects available GPUs and syncs accelerate & DeepSpeed configs

Usage:
    python setup_gpu_config.py [--gradient-accumulation-steps N]

Example:
    python setup_gpu_config.py --gradient-accumulation-steps 8
"""

import torch
import yaml
import json
import re
import argparse
from pathlib import Path


def extract_default_gradient_accumulation_steps(training_script="train_mpt7b_moe_accelerate.py"):
    """Extract the default gradient_accumulation_steps from the training script."""
    try:
        with open(training_script, 'r') as f:
            content = f.read()

        # Look for the argparse argument definition
        match = re.search(
            r'--gradient_accumulation_steps.*?default=(\d+)',
            content,
            re.DOTALL
        )
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"⚠ Could not extract gradient_accumulation_steps from {training_script}: {e}")

    return None


def update_accelerate_config(num_gpus):
    """Update accelerate config with detected GPU count."""
    config_file = Path("accelerate_config.yaml")
    if not config_file.exists():
        print(f"✗ Config file not found: {config_file}")
        return False

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update num_processes
    old_num = config.get('num_processes', 'unknown')
    config['num_processes'] = max(num_gpus, 1)  # At least 1 for CPU

    # Update distributed type based on GPU count
    if num_gpus > 1:
        config['distributed_type'] = 'DEEPSPEED'
    elif num_gpus == 1:
        config['distributed_type'] = 'NO'
    else:
        config['distributed_type'] = 'NO'
        config['use_cpu'] = True

    # Write updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated {config_file}")
    print(f"  - num_processes: {old_num} → {config['num_processes']}")
    print(f"  - distributed_type: {config['distributed_type']}")

    return True


def update_deepspeed_config(num_gpus, gradient_accumulation_steps):
    """Update DeepSpeed config to match training parameters."""
    config_file = Path("deepspeed_moe_config.json")
    if not config_file.exists():
        print(f"⚠ DeepSpeed config not found: {config_file}")
        return False

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Update gradient accumulation steps
    old_grad_acc = config.get('gradient_accumulation_steps', 'unknown')
    config['gradient_accumulation_steps'] = gradient_accumulation_steps

    # Get micro batch size
    micro_batch = config.get('train_micro_batch_size_per_gpu', 4)

    # Remove train_batch_size if present - let DeepSpeed calculate it
    if 'train_batch_size' in config:
        del config['train_batch_size']

    # Write updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')  # Add trailing newline

    print(f"\n✓ Updated {config_file}")
    print(f"  - gradient_accumulation_steps: {old_grad_acc} → {gradient_accumulation_steps}")
    print(f"  - train_batch_size: removed (auto-calculated)")

    # Calculate and display expected batch size
    if num_gpus > 1:
        total_batch_size = micro_batch * gradient_accumulation_steps * num_gpus
        print(f"\n📊 DeepSpeed will auto-calculate:")
        print(f"   train_batch_size = micro_batch × grad_acc_steps × num_gpus")
        print(f"                    = {micro_batch} × {gradient_accumulation_steps} × {num_gpus}")
        print(f"                    = {total_batch_size}")

    return True


def detect_and_update_configs(gradient_accumulation_steps=None):
    """Detect GPUs and update both accelerate and DeepSpeed configs."""

    # Detect number of GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ Detected {num_gpus} GPU(s):")
        for i in range(num_gpus):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        print("⚠ No CUDA GPUs detected. Will configure for CPU.")

    print()

    # Get gradient accumulation steps
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = extract_default_gradient_accumulation_steps()
        if gradient_accumulation_steps:
            print(f"✓ Using gradient_accumulation_steps from training script: {gradient_accumulation_steps}")
        else:
            gradient_accumulation_steps = 8  # Fallback default
            print(f"⚠ Using fallback gradient_accumulation_steps: {gradient_accumulation_steps}")
    else:
        print(f"✓ Using specified gradient_accumulation_steps: {gradient_accumulation_steps}")

    print()

    # Update both configs
    success = True
    success &= update_accelerate_config(num_gpus)

    if num_gpus > 1:  # Only update DeepSpeed config if using multiple GPUs
        success &= update_deepspeed_config(num_gpus, gradient_accumulation_steps)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically detect GPUs and sync training configs"
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=None,
        help='Override gradient accumulation steps (default: extract from training script)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GPU & Configuration Setup")
    print("=" * 60)
    print()

    success = detect_and_update_configs(args.gradient_accumulation_steps)

    if success:
        print("\n✓ All configurations updated successfully!")
        print("\nYou can now run training with:")
        print("  accelerate launch --config_file accelerate_config.yaml train_mpt7b_moe_accelerate.py")
    else:
        print("\n✗ Configuration update failed.")
        exit(1)
