#!/usr/bin/env python
"""Test script to verify GPU detection for multi-GPU training."""

import torch
from accelerate import Accelerator

def main():
    print("=" * 60)
    print("GPU Detection Test")
    print("=" * 60)

    # Test PyTorch CUDA detection
    print("\n1. PyTorch CUDA Detection:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"           Memory: {props.total_memory / 1024**3:.1f} GB")

    # Test Accelerate
    print("\n2. Accelerate Detection:")
    accelerator = Accelerator()
    print(f"   Device: {accelerator.device}")
    print(f"   Number of processes: {accelerator.num_processes}")
    print(f"   Is main process: {accelerator.is_main_process}")
    print(f"   Mixed precision: {accelerator.mixed_precision}")

    # Test simple tensor operation on GPU
    print("\n3. GPU Tensor Test:")
    try:
        x = torch.randn(1000, 1000).to(accelerator.device)
        y = torch.randn(1000, 1000).to(accelerator.device)
        z = torch.matmul(x, y)
        print(f"   ✓ Matrix multiplication successful on {z.device}")
        print(f"   Result shape: {z.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Setup Status:")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("✓ GPU setup is working correctly!")
        print(f"✓ Ready for multi-GPU training with {torch.cuda.device_count()} GPUs")
    else:
        print("✗ GPU setup has issues")
    print("=" * 60)

if __name__ == "__main__":
    main()
