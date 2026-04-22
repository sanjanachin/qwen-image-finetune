"""
Merge LoRA weights into the base Qwen-Image-Edit model.

After SFT training, this script:
  1. Loads the base Qwen-Image-Edit transformer
  2. Loads the LoRA adapter from a training checkpoint
  3. Merges LoRA into the base weights
  4. Saves as a full diffusers-compatible model directory

The output format is compatible with the counting-grpo RL pipeline:
    QwenImageTransformer2DModel.from_pretrained(output_dir, subfolder="transformer")

Usage:
    python scripts/counting/merge_lora.py \\
        --checkpoint outputs/counting_lora/counting_qwen_image_edit/v0/checkpoint-2-26430 \\
        --output-dir outputs/counting_merged

    # Copy full pipeline (VAE, tokenizer, scheduler) for standalone use:
    python scripts/counting/merge_lora.py \\
        --checkpoint outputs/counting_lora/counting_qwen_image_edit/v0/checkpoint-2-26430 \\
        --output-dir outputs/counting_merged \\
        --copy-pipeline
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def find_lora_weights(checkpoint_dir: str) -> str:
    """Find the LoRA safetensors file in a checkpoint directory."""
    candidates = [
        os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors"),
        os.path.join(checkpoint_dir, "adapter_model.safetensors"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No LoRA weights found in {checkpoint_dir}. "
        f"Expected one of: {[os.path.basename(c) for c in candidates]}"
    )


def merge_lora(
    base_model: str,
    checkpoint_dir: str,
    output_dir: str,
    adapter_name: str = "lora_edit",
    copy_pipeline: bool = False,
):
    from qflux.models.load_model import load_transformer

    logger.info("Loading base transformer from %s...", base_model)
    transformer = load_transformer(base_model, weight_dtype=torch.bfloat16)
    transformer = transformer.to("cpu")

    lora_path = find_lora_weights(checkpoint_dir)
    logger.info("Loading LoRA adapter from %s...", lora_path)
    transformer.load_lora_adapter(lora_path, adapter_name=adapter_name)
    transformer.set_adapter(adapter_name)

    lora_params = sum(1 for n, _ in transformer.named_parameters() if "lora" in n)
    logger.info("Loaded %d LoRA parameter tensors", lora_params)

    logger.info("Merging LoRA into base weights...")
    transformer.merge_adapter()
    transformer.unload_lora()
    logger.info("LoRA merged and unloaded")

    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)

    logger.info("Saving merged transformer to %s/...", transformer_dir)
    transformer.save_pretrained(transformer_dir)
    logger.info("Transformer saved")

    if copy_pipeline:
        logger.info("Copying pipeline components from base model...")
        from huggingface_hub import snapshot_download

        base_path = snapshot_download(base_model, allow_patterns=["*.json", "*.txt"])

        for item in ["vae", "scheduler", "tokenizer", "model_index.json"]:
            src = os.path.join(base_path, item)
            dst = os.path.join(output_dir, item)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                logger.info("  Copied %s/", item)
            elif os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                logger.info("  Copied %s", item)

    metadata = {
        "base_model": base_model,
        "checkpoint": checkpoint_dir,
        "lora_file": lora_path,
        "adapter_name": adapter_name,
        "dtype": "bfloat16",
    }
    meta_path = os.path.join(output_dir, "merge_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Merge complete. Output at: %s", output_dir)
    logger.info("To use with counting-grpo:")
    logger.info("  --pretrained_model_name_or_path %s", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA into base Qwen-Image-Edit for counting-grpo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint containing pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the merged model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="lora_edit",
        help="LoRA adapter name (must match training config)",
    )
    parser.add_argument(
        "--copy-pipeline",
        action="store_true",
        help="Copy VAE, scheduler, tokenizer from base model for standalone use",
    )
    args = parser.parse_args()

    merge_lora(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        adapter_name=args.adapter_name,
        copy_pipeline=args.copy_pipeline,
    )


if __name__ == "__main__":
    main()
