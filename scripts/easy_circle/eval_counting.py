"""
Evaluation script for counting task: Base Model vs Fine-tuned Model

This script evaluates whether fine-tuning improves counting accuracy by comparing
the base FLUX Kontext model against a fine-tuned checkpoint.

Metrics:
- Accuracy: % of images with exact count match
- Mean Absolute Error (MAE): Average |requested - detected|
- Median Absolute Error (MedAE): Median |requested - detected|

Usage:
    python eval_counting.py --checkpoint /path/to/checkpoint --num_samples 100
    python eval_counting.py --checkpoint outputs/.../checkpoint-0-17500 --num_samples 1500
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add SAM3 repo to path
SAM3_PATH = "/home/ubuntu/sanjana-fs/sam3"
if os.path.exists(SAM3_PATH):
    sys.path.insert(0, SAM3_PATH)
else:
    raise FileNotFoundError(f"SAM3 repository not found at {SAM3_PATH}. Please clone it from https://github.com/facebookresearch/sam3")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def initialize_sam3(device: str = "cuda:0", confidence_threshold: float = 0.3) -> Sam3Processor:
    """
    Initialize SAM3 model for dot detection.
    
    Args:
        device: Device to load model on
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Initialized SAM3 processor
    """
    print(f"Loading SAM3 model on {device}...")
    
    # Enable optimizations for GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Use bfloat16 for efficiency
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    # Build model with BPE vocabulary
    bpe_path = os.path.join(SAM3_PATH, "assets/bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        raise FileNotFoundError(f"BPE vocabulary file not found at {bpe_path}")
    
    model = build_sam3_image_model(bpe_path=bpe_path)
    model = model.to(device)
    
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("✅ SAM3 model loaded")
    return processor


def count_dots(
    image: Image.Image,
    processor: Sam3Processor,
    prompt: str = "red dots",
    sam3_device: str = "cuda:0"
) -> int:
    """
    Count the number of red dots in a generated image using SAM3.
    
    Uses SAM3 (Segment Anything Model 3) with text prompting to detect
    and count red dots in the image.
    
    Args:
        image: PIL Image to analyze
        processor: Initialized SAM3 processor (with confidence_threshold already set)
        prompt: Text prompt for detection (default: "red dots")
        sam3_device: Device SAM3 is on (not used, for API compatibility)
        
    Returns:
        Number of detected red dots
    """
    try:
        # Reset any previous prompts and set image
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        
        # Set text prompt
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        # Count detected objects from inference state
        # The inference_state contains 'masks', 'boxes', 'scores' after prompting
        if hasattr(inference_state, 'masks') and inference_state.masks is not None:
            count = len(inference_state.masks)
        elif 'masks' in inference_state and inference_state['masks'] is not None:
            count = len(inference_state['masks'])
        else:
            # Fallback: check obj_ids which represents detected objects
            if hasattr(inference_state, 'obj_ids'):
                count = len(inference_state.obj_ids)
            else:
                count = 0
        
        return count
            
    except Exception as e:
        print(f"Warning: Error in count_dots: {e}")
        import traceback
        traceback.print_exc()
        return 0


def extract_requested_count(prompt: str) -> int:
    """
    Extract the requested number of dots from the prompt.
    
    Expects format: "add {N} red dots to the image"
    
    Args:
        prompt: Text prompt string
        
    Returns:
        Requested number of dots
    """
    match = re.search(r'add (\d+) red dots', prompt.lower())
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not parse count from prompt: {prompt}")


def load_test_dataset(dataset_path: Path, num_samples: int = None) -> List[Tuple[Image.Image, str, int]]:
    """
    Load test dataset samples.
    
    Args:
        dataset_path: Path to test dataset directory (e.g., data/easy_circle/test)
        num_samples: Number of samples to load (None = all)
        
    Returns:
        List of (control_image, prompt, requested_count) tuples
    """
    control_dir = dataset_path / "control_images"
    training_dir = dataset_path / "training_images"
    
    # Get all prompt files
    prompt_files = sorted(training_dir.glob("*.txt"))
    
    if num_samples is not None:
        prompt_files = prompt_files[:num_samples]
    
    samples = []
    for prompt_file in tqdm(prompt_files, desc="Loading test samples"):
        # Read prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        # Get corresponding control image
        image_name = prompt_file.stem + ".png"
        control_img_path = control_dir / image_name
        
        if not control_img_path.exists():
            print(f"Warning: Missing control image for {prompt_file}, skipping...")
            continue
        
        control_img = Image.open(control_img_path).convert('RGB')
        requested_count = extract_requested_count(prompt)
        
        samples.append((control_img, prompt, requested_count))
    
    return samples


def run_inference(
    trainer: FluxKontextLoraTrainer,
    control_image: Image.Image,
    prompt: str,
    num_inference_steps: int = 20,
    cfg_scale: float = 3.5
) -> Image.Image:
    """
    Run inference with a trainer.
    
    Args:
        trainer: Initialized trainer in predict mode
        control_image: Input control image
        prompt: Text prompt
        num_inference_steps: Number of diffusion steps
        cfg_scale: Classifier-free guidance scale
        
    Returns:
        Generated PIL Image
    """
    result = trainer.predict(
        image=control_image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=cfg_scale,
        height=512,
        width=512
    )
    return result[0]


def calculate_metrics(requested: List[int], detected: List[int]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        requested: List of requested counts
        detected: List of detected counts
        
    Returns:
        Dictionary with metrics
    """
    requested = np.array(requested)
    detected = np.array(detected)
    
    # Absolute errors
    errors = np.abs(requested - detected)
    
    # Accuracy: exact match
    accuracy = np.mean(requested == detected) * 100.0
    
    # Mean Absolute Error
    mae = np.mean(errors)
    
    # Median Absolute Error
    medae = np.median(errors)
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'medae': medae
    }


def print_report(base_metrics: Dict[str, float], finetuned_metrics: Dict[str, float], num_samples: int):
    """
    Print evaluation report.
    
    Args:
        base_metrics: Metrics for base model
        finetuned_metrics: Metrics for fine-tuned model
        num_samples: Number of test samples
    """
    print("\n" + "=" * 80)
    print("Easy Circle Counting Evaluation")
    print("=" * 80)
    print(f"Test Set: {num_samples} images")
    print()
    print(f"{'Metric':<30} {'Base Model':<15} {'Fine-tuned':<15} {'Δ':<15}")
    print("-" * 80)
    
    # Accuracy
    base_acc = base_metrics['accuracy']
    ft_acc = finetuned_metrics['accuracy']
    delta_acc = ft_acc - base_acc
    print(f"{'Accuracy (exact match)':<30} {base_acc:>6.1f}%{'':<8} {ft_acc:>6.1f}%{'':<8} {delta_acc:>+6.1f} pp")
    
    # MAE
    base_mae = base_metrics['mae']
    ft_mae = finetuned_metrics['mae']
    delta_mae = ft_mae - base_mae
    print(f"{'Mean Absolute Error':<30} {base_mae:>6.2f} dots{'':<4} {ft_mae:>6.2f} dots{'':<4} {delta_mae:>+6.2f} dots")
    
    # MedAE
    base_medae = base_metrics['medae']
    ft_medae = finetuned_metrics['medae']
    delta_medae = ft_medae - base_medae
    print(f"{'Median Absolute Error':<30} {base_medae:>6.1f} dots{'':<4} {ft_medae:>6.1f} dots{'':<4} {delta_medae:>+6.1f} dots")
    
    print("=" * 80)
    print()
    
    # One-line summary
    print("Summary:")
    if ft_acc > base_acc:
        improvement = "improved"
    elif ft_acc < base_acc:
        improvement = "decreased"
    else:
        improvement = "maintained"
    
    print(f"Fine-tuning {improvement} exact counting accuracy from {base_acc:.1f}% to {ft_acc:.1f}%, "
          f"reducing average error from {base_mae:.2f} to {ft_mae:.2f} dots.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate counting accuracy: Base vs Fine-tuned model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint directory"
    )
    
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="/home/ubuntu/sanjana-fs/qwen-image-finetune/data/easy_circle/test",
        help="Path to test dataset directory"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (None = all)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ubuntu/sanjana-fs/qwen-image-finetune/configs/easy_circle_flux_kontext.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of diffusion steps"
    )
    
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale"
    )
    
    parser.add_argument(
        "--sam3_device",
        type=str,
        default="cuda:0",
        help="Device for SAM3 model (default: cuda:0)"
    )
    
    parser.add_argument(
        "--sam3_score_threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score for SAM3 detections (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    lora_weights = checkpoint_path / "pytorch_lora_weights.safetensors"
    if not lora_weights.exists():
        print(f"❌ Error: LoRA weights not found at {lora_weights}")
        sys.exit(1)
    
    test_dataset_path = Path(args.test_dataset)
    if not test_dataset_path.exists():
        print(f"❌ Error: Test dataset not found at {test_dataset_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Easy Circle Counting Evaluation: Base vs Fine-tuned")
    print("=" * 80)
    print(f"Fine-tuned checkpoint: {checkpoint_path}")
    print(f"Test dataset: {test_dataset_path}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print("=" * 80)
    print()
    
    # Load test dataset
    print("Step 1: Loading test dataset...")
    test_samples = load_test_dataset(test_dataset_path, args.num_samples)
    print(f"✅ Loaded {len(test_samples)} test samples")
    print()
    
    # Initialize SAM3 for dot detection
    print("Step 2: Initializing SAM3 for dot detection...")
    sam3_processor = initialize_sam3(args.sam3_device, confidence_threshold=args.sam3_score_threshold)
    print()
    
    # Process base model first (to manage memory)
    print("Step 3: Evaluating BASE model...")
    print("Initializing base model...")
    config_base = load_config_from_yaml(args.config)
    config_base.model.lora.pretrained_weight = None  # No LoRA weights
    trainer_base = FluxKontextLoraTrainer(config_base)
    trainer_base.setup_predict()
    print("✅ Base model ready")
    print()
    
    base_requested = []
    base_detected = []
    
    for control_img, prompt, requested_count in tqdm(test_samples, desc="Base model inference"):
        base_output = run_inference(
            trainer_base, 
            control_img, 
            prompt,
            args.num_inference_steps,
            args.cfg_scale
        )
        base_count = count_dots(
            base_output, 
            sam3_processor,
            prompt="red dots",
            sam3_device=args.sam3_device
        )
        base_requested.append(requested_count)
        base_detected.append(base_count)
    
    # Clean up base model to free memory
    print("\n🧹 Cleaning up base model...")
    del trainer_base
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Base model evaluation complete")
    print()
    
    # Process fine-tuned model
    print("Step 4: Evaluating FINE-TUNED model...")
    print("Initializing fine-tuned model...")
    config_ft = load_config_from_yaml(args.config)
    config_ft.model.lora.pretrained_weight = str(lora_weights)
    trainer_ft = FluxKontextLoraTrainer(config_ft)
    trainer_ft.setup_predict()
    print("✅ Fine-tuned model ready")
    print()
    
    ft_requested = []
    ft_detected = []
    
    for control_img, prompt, requested_count in tqdm(test_samples, desc="Fine-tuned model inference"):
        ft_output = run_inference(
            trainer_ft,
            control_img,
            prompt,
            args.num_inference_steps,
            args.cfg_scale
        )
        ft_count = count_dots(
            ft_output,
            sam3_processor,
            prompt="red dots",
            sam3_device=args.sam3_device
        )
        ft_requested.append(requested_count)
        ft_detected.append(ft_count)
    
    print("\n✅ Fine-tuned model evaluation complete")
    print()
    
    # Calculate metrics
    print("Step 5: Calculating metrics...")
    base_metrics = calculate_metrics(base_requested, base_detected)
    ft_metrics = calculate_metrics(ft_requested, ft_detected)
    print("✅ Metrics calculated")
    
    # Print report
    print_report(base_metrics, ft_metrics, len(test_samples))


if __name__ == "__main__":
    main()

