# Easy Circle Dataset Generator

This script generates a synthetic dataset for fine-tuning image-to-image models on the task of adding red dots to blank white images.

## Purpose

The dataset is designed to test whether fine-tuning can improve a model's ability to:
- Count accurately (generate the exact number of dots requested)
- Follow spatial instructions precisely
- Perform simple compositional tasks

## Dataset Structure

The generated dataset follows the repository's local directory structure format:

```
data/easy_circle/
├── train/
│   ├── control_images/     # Blank white 512x512 images (input)
│   │   ├── sample_00000.png
│   │   ├── sample_00001.png
│   │   └── ...
│   └── training_images/    # Images with red dots (target) + prompts
│       ├── sample_00000.png
│       ├── sample_00000.txt (e.g., "add 5 red dots to the image")
│       ├── sample_00001.png
│       ├── sample_00001.txt
│       └── ...
├── val/
│   ├── control_images/
│   └── training_images/
└── test/
    ├── control_images/
    └── training_images/
```

## Usage

### Basic Usage

```bash
python scripts/easy_circle/easy_circle_datagen.py \
    --max_num_dots 40 \
    --num_samples 1000 \
    --num_pixels 15 \
    --overlap
```

### Command Line Arguments

**Required Arguments:**
- `--max_num_dots`: Maximum number of dots in any image (range will be 0 to this value)
- `--num_samples`: Total number of samples to generate (automatically split 70/15/15 into train/val/test)
- `--num_pixels`: Radius of each dot in pixels

**Optional Arguments:**
- `--overlap`: Flag to allow dots to overlap (default: False - dots cannot overlap)
- `--output_dir`: Output directory path (default: `data/easy_circle`)
- `--seed`: Random seed for reproducibility (default: 42)

### Examples

#### Small dataset for quick testing (with overlapping dots)
```bash
python scripts/easy_circle/easy_circle_datagen.py \
    --max_num_dots 20 \
    --num_samples 100 \
    --num_pixels 10 \
    --overlap
```

#### Medium dataset for actual training (no overlapping)
```bash
python scripts/easy_circle/easy_circle_datagen.py \
    --max_num_dots 40 \
    --num_samples 1000 \
    --num_pixels 15
```

#### Large dataset with larger dots
```bash
python scripts/easy_circle/easy_circle_datagen.py \
    --max_num_dots 50 \
    --num_samples 5000 \
    --num_pixels 20 \
    --overlap
```

## Dataset Specifications

- **Image Size**: 512x512 pixels
- **Background**: Pure white RGB(255, 255, 255)
- **Dot Color**: Pure red RGB(255, 0, 0)
- **Dot Count Distribution**: Uniform distribution from 0 to max_num_dots
- **Data Split**: 70% train / 15% val / 15% test
- **Prompt Format**: `"add {N} red dots to the image"` where N is the actual number of dots

## Assumptions and Implementation Notes

1. **Overlap Handling**: 
   - When `--overlap` is used, dots can overlap freely
   - When overlap is disabled, the script attempts up to 1000 placements per dot
   - If placement fails (canvas too crowded), a warning is printed

2. **Dot Placement**: 
   - Dots are placed at uniformly random positions
   - Dot centers are constrained to be at least `radius` pixels from image edges
   - This ensures complete dots are visible (no partial dots at boundaries)

3. **Reproducibility**: 
   - Use `--seed` parameter to ensure consistent dataset generation
   - Default seed is 42

4. **File Naming**: 
   - Samples are named sequentially: `sample_00000`, `sample_00001`, etc.
   - Each sample has 3 files: control image (.png), target image (.png), and prompt (.txt)

## Integration with Training

After generating the dataset, you can use it with the training configs by setting:

```yaml
data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path:
      - data/easy_circle/train  # For training
    # ... other config options
```

For validation/testing, point to `data/easy_circle/val` or `data/easy_circle/test`.

## Dependencies

The script requires:
- `Pillow` (PIL) - for image generation and drawing
- `numpy` - for numerical operations
- `tqdm` - for progress bars

These should already be installed if you have the repository dependencies installed.

