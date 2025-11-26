"""
Easy Circle Dataset Generator

Generates a synthetic dataset for fine-tuning image-to-image models on a task
of adding red dots to blank white images. The dataset follows the local directory
structure format expected by this repository.

Dataset Structure:
    data/easy_circle/
    ├── train/
    │   ├── control_images/     # Blank white images (input)
    │   └── training_images/    # Images with red dots (target) + prompts
    ├── val/
    │   ├── control_images/
    │   └── training_images/
    └── test/
        ├── control_images/
        └── training_images/

Usage:
    python scripts/easy_circle/easy_circle_datagen.py \\
        --max_num_dots 40 \\
        --num_samples 1000 \\
        --num_pixels 15 \\
        --overlap

Assumptions:
    - Image size: 512x512 pixels
    - Red color: RGB(255, 0, 0)
    - White background: RGB(255, 255, 255)
    - Train/Val/Test split: 70%/15%/15%
    - Uniform distribution of dot counts (0 to max_num_dots)
    - Dots are placed with uniform random positions
    - When overlap=False, script attempts placement up to 1000 times per dot
      before giving up (to handle cases where canvas is too full)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def generate_blank_image(width: int = 512, height: int = 512) -> Image.Image:
    """Generate a blank white image.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        PIL Image with white background
    """
    return Image.new('RGB', (width, height), color=(255, 255, 255))


def check_overlap(
    new_center: Tuple[int, int],
    existing_centers: List[Tuple[int, int]],
    radius: int
) -> bool:
    """Check if a new circle overlaps with existing circles.
    
    Args:
        new_center: (x, y) coordinates of the new circle center
        existing_centers: List of (x, y) coordinates of existing circles
        radius: Radius of circles in pixels
    
    Returns:
        True if there is overlap, False otherwise
    """
    new_x, new_y = new_center
    for ex_x, ex_y in existing_centers:
        # Calculate distance between centers
        distance = np.sqrt((new_x - ex_x) ** 2 + (new_y - ex_y) ** 2)
        # Circles overlap if distance < 2*radius
        if distance < 2 * radius:
            return True
    return False


def add_dots_to_image(
    image: Image.Image,
    num_dots: int,
    radius: int,
    allow_overlap: bool = True,
    max_attempts: int = 1000
) -> Image.Image:
    """Add red dots to an image.
    
    Args:
        image: PIL Image to draw on
        num_dots: Number of red dots to add
        radius: Radius of each dot in pixels
        allow_overlap: Whether dots can overlap
        max_attempts: Maximum placement attempts per dot (for non-overlapping mode)
    
    Returns:
        PIL Image with red dots drawn on it
    
    Note:
        In non-overlapping mode, if a dot cannot be placed after max_attempts,
        it will be skipped. This can happen when the canvas is too crowded.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Red color
    red_color = (255, 0, 0)
    
    # Track centers if we need to check for overlap
    centers = []
    
    dots_placed = 0
    for _ in range(num_dots):
        placed = False
        
        for attempt in range(max_attempts):
            # Random center position (ensure dot stays within image bounds)
            center_x = random.randint(radius, width - radius)
            center_y = random.randint(radius, height - radius)
            
            # Check overlap if needed
            if not allow_overlap:
                if check_overlap((center_x, center_y), centers, radius):
                    continue  # Try again
            
            # Draw the dot (filled circle)
            bbox = [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius
            ]
            draw.ellipse(bbox, fill=red_color, outline=red_color)
            
            # Record center
            centers.append((center_x, center_y))
            dots_placed += 1
            placed = True
            break
        
        if not placed and not allow_overlap:
            print(f"Warning: Could not place dot {dots_placed + 1}/{num_dots} "
                  f"after {max_attempts} attempts. Canvas may be too crowded.")
    
    return image


def generate_sample(
    sample_id: int,
    num_dots: int,
    radius: int,
    allow_overlap: bool,
    output_dir: Path,
    split: str
) -> None:
    """Generate a single sample (control image, target image, and prompt).
    
    Args:
        sample_id: Unique identifier for this sample
        num_dots: Number of dots to add
        radius: Radius of each dot
        allow_overlap: Whether dots can overlap
        output_dir: Root output directory (e.g., data/easy_circle)
        split: Dataset split ('train', 'val', or 'test')
    """
    # Create sample filename
    sample_name = f"sample_{sample_id:05d}"
    
    # Paths
    control_dir = output_dir / split / "control_images"
    training_dir = output_dir / split / "training_images"
    
    control_img_path = control_dir / f"{sample_name}.png"
    target_img_path = training_dir / f"{sample_name}.png"
    prompt_path = training_dir / f"{sample_name}.txt"
    
    # Generate control image (blank white)
    control_img = generate_blank_image()
    control_img.save(control_img_path)
    
    # Generate target image (with dots)
    target_img = generate_blank_image()
    target_img = add_dots_to_image(target_img, num_dots, radius, allow_overlap)
    target_img.save(target_img_path)
    
    # Generate prompt
    prompt = f"add {num_dots} red dots to the image"
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)


def create_directory_structure(output_dir: Path) -> None:
    """Create the dataset directory structure.
    
    Args:
        output_dir: Root directory for the dataset
    """
    splits = ['train', 'val', 'test']
    subdirs = ['control_images', 'training_images']
    
    for split in splits:
        for subdir in subdirs:
            dir_path = output_dir / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {output_dir}")


def generate_dataset(
    max_num_dots: int,
    num_samples: int,
    num_pixels: int,
    overlap: bool,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """Generate the complete dataset.
    
    Args:
        max_num_dots: Maximum number of dots in any image (0 to max_num_dots)
        num_samples: Total number of samples to generate
        num_pixels: Radius of each dot in pixels
        overlap: Whether dots can overlap
        output_dir: Root output directory
        train_ratio: Proportion of samples for training
        val_ratio: Proportion of samples for validation
        test_ratio: Proportion of samples for testing
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Calculate split sizes
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size  # Remainder goes to test
    
    print(f"\nDataset Configuration:")
    print(f"  Total samples: {num_samples}")
    print(f"  Train: {train_size} ({train_ratio*100:.1f}%)")
    print(f"  Val: {val_size} ({val_ratio*100:.1f}%)")
    print(f"  Test: {test_size} ({test_ratio*100:.1f}%)")
    print(f"  Dot range: 0 to {max_num_dots}")
    print(f"  Dot radius: {num_pixels} pixels")
    print(f"  Overlap allowed: {overlap}")
    print(f"  Random seed: {seed}\n")
    
    # Generate samples for each split
    sample_id = 0
    
    for split, split_size in [('train', train_size), ('val', val_size), ('test', test_size)]:
        print(f"Generating {split} set ({split_size} samples)...")
        
        for _ in tqdm(range(split_size), desc=f"  {split}"):
            # Randomly select number of dots (uniform distribution)
            num_dots = random.randint(0, max_num_dots)
            
            # Generate the sample
            generate_sample(
                sample_id=sample_id,
                num_dots=num_dots,
                radius=num_pixels,
                allow_overlap=overlap,
                output_dir=output_dir,
                split=split
            )
            
            sample_id += 1
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Dataset saved to: {output_dir}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── control_images/  ({train_size} blank images)")
    print(f"    │   └── training_images/ ({train_size} target images + prompts)")
    print(f"    ├── val/")
    print(f"    │   ├── control_images/  ({val_size} blank images)")
    print(f"    │   └── training_images/ ({val_size} target images + prompts)")
    print(f"    └── test/")
    print(f"        ├── control_images/  ({test_size} blank images)")
    print(f"        └── training_images/ ({test_size} target images + prompts)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset of images with red dots for fine-tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--max_num_dots',
        type=int,
        required=True,
        help='Maximum number of dots in any image (range: 0 to max_num_dots)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        required=True,
        help='Total number of samples to generate (will be split into train/val/test)'
    )
    
    parser.add_argument(
        '--num_pixels',
        type=int,
        required=True,
        help='Radius of each dot in pixels'
    )
    
    parser.add_argument(
        '--overlap',
        action='store_true',
        help='Allow dots to overlap (default: False - dots cannot overlap)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/easy_circle',
        help='Output directory for the dataset (relative to repo root)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Get repository root (assuming script is in scripts/easy_circle/)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent
    output_path = repo_root / args.output_dir
    
    print("=" * 60)
    print("Easy Circle Dataset Generator")
    print("=" * 60)
    
    # Generate dataset
    generate_dataset(
        max_num_dots=args.max_num_dots,
        num_samples=args.num_samples,
        num_pixels=args.num_pixels,
        overlap=args.overlap,
        output_dir=output_path,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

