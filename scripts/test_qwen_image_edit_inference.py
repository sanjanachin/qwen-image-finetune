"""
Test inference script for Qwen-Image-Edit base model.

Usage:
    cd /home/ubuntu/sanjana-fs-us-east-1/qwen-image-finetune
    python scripts/test_qwen_image_edit_inference.py \
        --image path/to/source.png \
        --prompt "add 5 birds to the tree"

    # With LoRA checkpoint:
    python scripts/test_qwen_image_edit_inference.py \
        --image path/to/source.png \
        --prompt "add 5 birds to the tree" \
        --lora_weights path/to/pytorch_lora_weights.safetensors

    # Custom output size:
    python scripts/test_qwen_image_edit_inference.py \
        --image path/to/source.png \
        --prompt "add 5 birds to the tree" \
        --height 1024 --width 1024
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

import torch
from PIL import Image

from qflux.data.config import Config, ModelConfig, LoraConfig, PredictConfig, DeviceConfig


def build_minimal_config(
    model_path: str = "Qwen/Qwen-Image-Edit",
    lora_weights: str | None = None,
) -> Config:
    """Build a minimal Config object for inference programmatically."""
    config = Config(
        trainer="QwenImageEdit",
        mode="predict",
        model=ModelConfig(
            pretrained_model_name_or_path=model_path,
            quantize=False,
            lora=LoraConfig(
                pretrained_weight=lora_weights,
            ),
        ),
        predict=PredictConfig(
            devices=DeviceConfig(
                vae="cuda:0",
                text_encoder="cuda:0",
                dit="cuda:0",
            ),
        ),
    )
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen-Image-Edit inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to source image")
    parser.add_argument("--prompt", type=str, required=True, help="Editing instruction")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights (.safetensors)")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--height", type=int, default=None, help="Output height (defaults to source image height)")
    parser.add_argument("--width", type=int, default=None, help="Output width (defaults to source image width)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Source image not found: {args.image}")
        sys.exit(1)

    source_image = Image.open(args.image).convert("RGB")
    print(f"Source image: {args.image} ({source_image.size[0]}x{source_image.size[1]})")
    print(f"Prompt: \"{args.prompt}\"")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seed: {args.seed}")

    config = build_minimal_config(
        model_path=args.model_path,
        lora_weights=args.lora_weights,
    )

    print(f"\nLoading model: {args.model_path}")
    if args.lora_weights:
        print(f"LoRA weights: {args.lora_weights}")

    from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

    trainer = QwenImageEditTrainer(config)

    height = args.height or source_image.size[1]
    width = args.width or source_image.size[0]
    # Make divisible by 16
    height = (height // 16) * 16
    width = (width // 16) * 16
    print(f"Output dimensions: {width}x{height}")

    import time

    print(f"\nRunning inference ({args.num_inference_steps} steps, cfg={args.true_cfg_scale})...")
    start_time = time.time()
    results = trainer.predict(
        image=source_image,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        weight_dtype=torch.bfloat16,
        height=height,
        width=width,
        output_type="pil",
    )
    elapsed = time.time() - start_time

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results[0].save(args.output)
    print(f"\nSaved to: {args.output}")
    print(f"Inference time: {elapsed:.1f}s ({elapsed / args.num_inference_steps:.2f}s/step)")


if __name__ == "__main__":
    main()
