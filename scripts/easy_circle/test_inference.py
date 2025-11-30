"""
Simple inference script to test trained Easy Circle LoRA model.

Usage:
    python test_inference.py --checkpoint /path/to/checkpoint --num_dots 10

This script:
1. Loads the trained LoRA weights
2. Creates a blank white image
3. Generates an image with N red dots
4. Saves the result
"""

import argparse
from pathlib import Path
from PIL import Image
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Test Easy Circle LoRA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., outputs/easy_circle_lora/checkpoint-3000)",
    )
    parser.add_argument(
        "--num_dots",
        type=int,
        default=10,
        help="Number of red dots to generate (0-20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale",
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    lora_weights = checkpoint_path / "pytorch_lora_weights.safetensors"
    if not lora_weights.exists():
        print(f"❌ Error: LoRA weights not found at {lora_weights}")
        return
    
    print("=" * 60)
    print("Easy Circle Inference")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt: add {args.num_dots} red dots to the image")
    print(f"Inference steps: {args.steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print("=" * 60)
    print()
    
    # Load configuration
    print("📦 Loading configuration...")
    config_path = Path(__file__).parent.parent.parent / "configs" / "easy_circle_flux_kontext.yaml"
    config = load_config_from_yaml(str(config_path))
    
    # Set LoRA weights path
    config.model.lora.pretrained_weight = str(lora_weights)
    
    # Initialize trainer
    print("🔧 Initializing trainer...")
    trainer = FluxKontextLoraTrainer(config)
    
    # Setup for prediction
    print("⚙️  Setting up prediction mode...")
    trainer.setup_predict()
    
    # Create blank white image
    print("🖼️  Creating blank white image...")
    blank_image = Image.new('RGB', (512, 512), color=(255, 255, 255))
    
    # Generate prompt
    prompt = f"add {args.num_dots} red dots to the image"
    
    # Run inference
    print(f"🎨 Generating image with {args.num_dots} red dots...")
    result = trainer.predict(
        image=blank_image,
        prompt=prompt,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg_scale,
        height=512,
        width=512
    )
    
    # Save result
    output_path = Path(args.output)
    result[0].save(output_path)
    
    print()
    print("=" * 60)
    print(f"✅ Success! Image saved to: {output_path.absolute()}")
    print("=" * 60)
    
    # Display image if running in notebook or interactive environment
    try:
        from IPython.display import display
        display(result[0])
    except ImportError:
        pass


if __name__ == "__main__":
    main()

