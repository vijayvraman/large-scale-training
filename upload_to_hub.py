"""
Upload trained model to HuggingFace Hub

This script uploads your fine-tuned MPT-7B-MoE model to HuggingFace Hub.

Usage:
    python upload_to_hub.py \
        --model_path ./mpt7b_moe_finetune/checkpoint-step-2000 \
        --repo_name your-username/mpt-7b-moe-nq-finetuned \
        --private

Requirements:
    - huggingface_hub library: pip install huggingface-hub
    - HuggingFace account and login: huggingface-cli login
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, whoami
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory (e.g., ./mpt7b_moe_finetune/checkpoint-step-2000)"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on HuggingFace Hub (e.g., username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned MPT-7B-MoE model",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default=None,
        help="Path to custom README.md file (model card)"
    )
    return parser.parse_args()


def create_model_card(args, base_model: str = "mosaicml/mpt-7b") -> str:
    """Generate a basic model card if none provided."""

    card = f"""---
language: en
license: apache-2.0
tags:
- text-generation
- mpt
- moe
- question-answering
- natural-questions
base_model: {base_model}
---

# MPT-7B-MoE Fine-tuned on Natural Questions

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) on the Natural Questions dataset.

## Model Description

- **Base Model**: {base_model}
- **Architecture**: Mixture-of-Experts (MoE) variant
- **Training Dataset**: Natural Questions (NQ) - annotated for expert routing
- **Task**: Question Answering / Text Generation

## Training Details

The model was fine-tuned using:
- DeepSpeed with ZeRO optimization
- HuggingFace Accelerate
- Custom MoE expert annotations

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "{args.repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

# Example inference
question = "What is the capital of France?"
prompt = f"Question: {{question}}\\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Training Configuration

Training was performed using the configuration in the repository. See `train_mixtral_8x7b_moe_accelerate.py` for details.

## Limitations and Biases

This model inherits limitations and biases from the base MPT-7B model and the Natural Questions dataset.
Users should be aware of potential biases in question-answering outputs.

## Citation

If you use this model, please cite:

```bibtex
@misc{{mpt-7b-moe-nq,
  author = {{Your Name}},
  title = {{MPT-7B-MoE Fine-tuned on Natural Questions}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{args.repo_name}}}}}
}}
```

## Acknowledgements

- Base model by MosaicML
- Natural Questions dataset by Google Research
- Training infrastructure using DeepSpeed and HuggingFace Accelerate
"""
    return card


def verify_login():
    """Verify user is logged in to HuggingFace Hub."""
    try:
        user_info = whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        logger.error("Not logged in to HuggingFace Hub!")
        logger.error("Please run: huggingface-cli login")
        logger.error(f"Error: {e}")
        return False


def main():
    args = parse_args()

    # Verify model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return

    if not model_path.is_dir():
        logger.error(f"Model path is not a directory: {model_path}")
        return

    # Check for required files
    required_files = ["config.json", "model.safetensors.index.json"]
    for file in required_files:
        if not (model_path / file).exists():
            logger.error(f"Required file not found: {file}")
            return

    logger.info(f"Model path verified: {model_path}")
    logger.info(f"Files found: {list(model_path.glob('*'))}")

    # Verify HuggingFace login
    if not verify_login():
        return

    # Initialize HF API
    api = HfApi()

    # Create repository
    logger.info(f"Creating/accessing repository: {args.repo_name}")
    try:
        repo_url = create_repo(
            repo_id=args.repo_name,
            private=args.private,
            exist_ok=True,
            repo_type="model"
        )
        logger.info(f"Repository ready: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        return

    # Create or load model card
    if args.model_card and os.path.exists(args.model_card):
        logger.info(f"Using custom model card: {args.model_card}")
        with open(args.model_card, 'r') as f:
            model_card_content = f.read()
    else:
        logger.info("Generating default model card")
        model_card_content = create_model_card(args)

    # Save model card to model directory
    readme_path = model_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card_content)
    logger.info(f"Model card saved to: {readme_path}")

    # Upload folder to Hub
    logger.info("Starting upload to HuggingFace Hub...")
    logger.info("This may take a while depending on model size and connection speed...")

    try:
        result = upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_name,
            repo_type="model",
            commit_message=args.commit_message,
        )
        logger.info(f"Upload successful!")
        logger.info(f"Model URL: https://huggingface.co/{args.repo_name}")
        logger.info(f"Commit: {result}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return

    logger.info("=" * 60)
    logger.info("Upload complete! Your model is now available at:")
    logger.info(f"https://huggingface.co/{args.repo_name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
