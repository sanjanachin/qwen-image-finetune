# Easy Circle Fine-tuning Guide

This directory contains scripts and configurations for fine-tuning FLUX Kontext to add red dots to blank images.

## Quick Start

### 1. Generate Dataset (if not already done)

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune
python scripts/easy_circle/easy_circle_datagen.py \
    --max_num_dots 20 \
    --num_samples 10000 \
    --num_pixels 15
```

This creates:
- **7,000 training samples** (data/easy_circle/train/)
- **1,500 validation samples** (data/easy_circle/val/)
- **1,500 test samples** (data/easy_circle/test/)

### 2. Run Training

```bash
cd scripts/easy_circle
./train_easy_circle.sh
```

The script will:
1. Ask if you want to build cache (recommended: yes)
2. Start training with validation every 100 steps
3. Save checkpoints every 200 steps

**Expected Training Time**: 
- Cache building: ~10-15 minutes
- Training (3000 steps): ~2-3 hours on A100 40GB

### 3. Monitor Training

In a separate terminal, launch TensorBoard:

```bash
tensorboard --logdir=/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora
```

Then open http://localhost:6006 in your browser to see:
- Training loss curves
- Validation images generated every 100 steps
- Learning rate schedule

## Manual Training Steps

If you prefer manual control:

### Step 1: Build Cache (Optional but Recommended)

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/src
python -m qflux.main \
    --config ../configs/easy_circle_flux_kontext.yaml \
    --cache
```

### Step 2: Train

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/src
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file ../accelerate_config_single_gpu.yaml \
    -m qflux.main \
    --config ../configs/easy_circle_flux_kontext.yaml
```

### Step 3: Resume Training (if interrupted)

Edit `configs/easy_circle_flux_kontext.yaml` and set:
```yaml
resume: /home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora/checkpoint-XXXX
```

Then run the training command again.

## Output Structure

```
outputs/easy_circle_lora/
├── easy_circle_flux_kontext/
│   ├── checkpoint-200/          # Checkpoint at step 200
│   │   ├── pytorch_lora_weights.safetensors
│   │   └── optimizer.bin
│   ├── checkpoint-400/          # Checkpoint at step 400
│   ├── ...
│   └── cache/                   # Embedding cache (if built)
└── events.out.tfevents.*        # TensorBoard logs
```

## Inference After Training

Use the trained model:

```python
from qflux.trainer.flux_kontext_trainer import FluxKontextTrainer
from qflux.data.config import load_config_from_yaml
from PIL import Image

# Load config and specify LoRA weights
config = load_config_from_yaml("configs/easy_circle_flux_kontext.yaml")
config.model.lora.pretrained_weight = "outputs/easy_circle_lora/checkpoint-3000/pytorch_lora_weights.safetensors"

# Initialize trainer
trainer = FluxKontextTrainer(config)
trainer.setup_predict()

# Create blank white image
blank_image = Image.new('RGB', (512, 512), color=(255, 255, 255))

# Generate image with red dots
result = trainer.predict(
    prompt_image=blank_image,
    prompt="add 10 red dots to the image",
    num_inference_steps=20,
    true_cfg_scale=3.5
)

# Save result
result[0].save("output_with_dots.png")
```

## Configuration Details

Key parameters in `configs/easy_circle_flux_kontext.yaml`:

- **LoRA rank**: 16 (good balance of quality and memory)
- **Batch size**: 2 (optimal for A100 40GB)
- **Learning rate**: 0.0001 (conservative for quality)
- **Max steps**: 3000 (~4.3 epochs)
- **Mixed precision**: bf16 (best quality on A100)
- **Validation**: Every 100 steps with 4 samples
- **Checkpoints**: Every 200 steps, keep last 10

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1 in config
- Ensure `gradient_checkpointing: true`

### Training Too Slow
- Build cache first with `--cache` flag
- Ensure cache is being used (check logs)

### Validation Images Not Showing
- Check TensorBoard is pointing to correct log directory
- Wait until first validation step (step 100)

### Import Errors
- Always run commands from `src/` directory
- Check virtual environment is activated

## Next Steps

After training:
1. Compare checkpoints using validation metrics
2. Test on held-out test set
3. Share model on HuggingFace Hub (optional)
4. Experiment with different hyperparameters
