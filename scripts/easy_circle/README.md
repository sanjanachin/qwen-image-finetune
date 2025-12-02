# Easy Circle Fine-tuning Guide

This directory contains scripts and configurations for fine-tuning FLUX Kontext to add red dots to blank images.

## 🚀 Quick Start

### Step 1: Generate Dataset (if not already done)

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

### Step 2: Start Training

```bash
cd scripts/easy_circle
./train_easy_circle.sh
```

The script will:
1. Ask if you want to build cache (recommended: yes)
2. Start training with validation every 50 steps
3. Save checkpoints every 100 steps

### Step 3: Monitor Training

In a separate terminal, launch TensorBoard:

```bash
tensorboard --logdir=/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora
```

Then open http://localhost:6006 in your browser to see:
- Training loss curves
- Validation images generated every 50 steps
- Learning rate schedule

---

## 📐 Understanding Training Configuration

### Current Test Configuration

The config is set for a **quick test run** (300 steps, ~30 minutes):

```yaml
max_train_steps: 300  # TEST RUN: ~30 minutes
# num_epochs: 3       # FULL TRAINING: Uncomment for 3 full epochs
```

### Dataset Math

With your dataset:
- **Training samples**: 7,000
- **Batch size**: 2
- **Steps per epoch**: 7,000 ÷ 2 = 3,500 steps
- **Time per step**: ~3-6 seconds (depends on GPU)

### Training Time Estimates

| Configuration | Steps | Epochs | Time | Use Case |
|--------------|-------|--------|------|----------|
| Test run (current) | 300 | 0.09 | 30 min | Verify setup works |
| Quick training | 3,500 | 1 | ~6 hours | Simple tasks |
| Balanced (recommended) | 10,500 | 3 | ~18 hours | Quality results |
| Thorough | 17,500 | 5 | ~29 hours | Best quality |
| Very thorough | 35,000 | 10 | ~58 hours | Complex patterns |

---

## 🎯 Recommended Training Workflow

### Phase 1: Test Run (30 minutes) - DO THIS FIRST

Run the test configuration to verify everything works:

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/scripts/easy_circle
./train_easy_circle.sh
```

**What to check:**
- Does cache build successfully? (~10-15 minutes)
- Does training start without errors?
- Can you see validation images in TensorBoard?
- What's the actual training speed?
- GPU memory usage (should be ~30-35GB on A100 40GB)

### Phase 2: Analyze Test Results

Launch TensorBoard and review:
- **Training loss**: Should be decreasing
- **Validation images**: Check steps 50, 100, 150, 200, 250, 300
- **Training speed**: Note actual seconds per step

### Phase 3: Configure Full Training

Based on test results, edit `configs/easy_circle_flux_kontext.yaml`:

**Option 1: Quick Training (1 epoch)**
```yaml
train:
  # max_train_steps: 300  # Comment this out
  num_epochs: 1           # Add this
```

**Option 2: Balanced Training (3 epochs) - RECOMMENDED**
```yaml
train:
  # max_train_steps: 300  # Comment this out
  num_epochs: 3           # Add this
```

**Option 3: Thorough Training (5 epochs)**
```yaml
train:
  # max_train_steps: 300  # Comment this out
  num_epochs: 5           # Add this
```

### Phase 4: Resume for Full Training

The framework automatically continues from the last checkpoint! Just update the config and run:

```bash
./train_easy_circle.sh
```

Or explicitly specify resuming:
```yaml
resume: /home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora/easy_circle_flux_kontext/v0/checkpoint-0-300
```

---

## 🛠️ Manual Training Commands

If you prefer manual control over the training process:

### Build Cache (Optional but Recommended)

Pre-compute VAE and text encoder embeddings (makes training 2-3x faster):

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune
python -m qflux.main \
    --config configs/easy_circle_flux_kontext.yaml \
    --cache
```

**Cache building time**: ~10-15 minutes (one-time only)
- Processes all 7,000 training samples
- Saves VAE latents and text embeddings
- Cache is automatically used in subsequent training runs

### Start Training

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file accelerate_config_single_gpu.yaml \
    -m qflux.main \
    --config configs/easy_circle_flux_kontext.yaml
```

### Resume Training (if interrupted)

Edit `configs/easy_circle_flux_kontext.yaml`:
```yaml
resume: /path/to/checkpoint-XXXX
```

Then run the training command again. The trainer will:
- Load the saved LoRA weights
- Resume from the saved step number
- Continue with the same optimizer state

---

## 📁 Output Structure

```
outputs/easy_circle_lora/
├── cache/                              # Cached embeddings (if built)
│   ├── control_ids/
│   ├── control_latents/
│   ├── image_latents/
│   ├── prompt_embeds/
│   └── pooled_prompt_embeds/
└── easy_circle_flux_kontext/
    └── v0/                            # Version 0 (auto-increments)
        ├── checkpoint-0-100/          # Format: checkpoint-{epoch}-{step}
        │   ├── pytorch_lora_weights.safetensors
        │   └── state.json
        ├── checkpoint-0-200/
        ├── checkpoint-0-300/
        ├── ...
        ├── events.out.tfevents.*     # TensorBoard logs
        └── train_config.yaml         # Snapshot of config used
```

**Checkpoints contain:**
- `pytorch_lora_weights.safetensors` (~71 MB) - Trained LoRA weights
- `state.json` - Training state (step, epoch, is_last flag)

**TensorBoard logs include:**
- Training loss at each step
- Learning rate schedule
- Validation images at configured intervals
- System metrics (GPU, memory)

---

## ⚙️ Configuration Details

Key parameters in `configs/easy_circle_flux_kontext.yaml`:

### Model Configuration
```yaml
model:
  pretrained_model_name_or_path: lrzjason/flux-kontext-nf4
  quantize: false  # Already pre-quantized
  lora:
    r: 16          # LoRA rank (higher = more capacity, more memory)
    lora_alpha: 16 # Usually equal to rank
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
```

### Training Configuration
```yaml
data:
  batch_size: 2   # Optimal for A100 40GB

optimizer:
  class_path: bitsandbytes.optim.Adam8bit
  init_args:
    lr: 0.0001    # Conservative learning rate for quality

lr_scheduler:
  scheduler_type: "cosine"  # Smooth decay
  warmup_steps: 100

train:
  gradient_accumulation_steps: 1
  max_train_steps: 300        # For test run
  # num_epochs: 3             # Uncomment for full training
  checkpointing_steps: 100    # Save frequency
  checkpoints_total_limit: 5  # Keep only last 5 (saves disk space)
  mixed_precision: "bf16"     # Best quality on A100
  gradient_checkpointing: true

validation:
  enabled: true
  steps: 50          # Validate every 50 steps
  max_samples: 2     # Generate 2 validation images
  seed: 42           # Reproducible validation
```

---

## 🖼️ Inference After Training

### Using Trained Checkpoint

```python
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from PIL import Image

# Load config and specify LoRA weights
config = load_config_from_yaml("configs/easy_circle_flux_kontext.yaml")
config.model.lora.pretrained_weight = "outputs/easy_circle_lora/easy_circle_flux_kontext/v0/checkpoint-5-20100/pytorch_lora_weights.safetensors"

# Initialize trainer in predict mode
trainer = FluxKontextLoraTrainer(config)
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

### From HuggingFace Hub

If you uploaded your model to HuggingFace:

```python
from diffusers import FluxKontextPipeline

# Load base model
pipe = FluxKontextPipeline.from_pretrained("lrzjason/flux-kontext-nf4")

# Load your trained LoRA weights
pipe.load_lora_weights(
    "your-username/easy-circle-flux-lora",
    subfolder="checkpoint-5-20100"
)

# Generate
blank_image = Image.new('RGB', (512, 512), color=(255, 255, 255))
output = pipe(
    prompt="add 10 red dots",
    image=blank_image,
    num_inference_steps=20
)
output[0].save("result.png")
```

---

## 🔧 Troubleshooting

### Out of Memory Errors

**Symptoms**: CUDA out of memory during training

**Solutions**:
1. Reduce `batch_size` to 1 in config
2. Ensure `gradient_checkpointing: true`
3. Enable `low_memory: true` in train config
4. Use smaller LoRA rank (try `r: 8`)

```yaml
data:
  batch_size: 1  # Reduce from 2

train:
  gradient_checkpointing: true
  low_memory: true

model:
  lora:
    r: 8  # Reduce from 16
```

### Training Too Slow

**Symptoms**: More than 10 seconds per step

**Solutions**:
1. Build cache first with `--cache` flag
2. Verify cache is being used (check logs for "Loading from cache")
3. Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check GPU utilization: `nvidia-smi`

### Validation Images Not Showing in TensorBoard

**Symptoms**: Loss curves visible but no images

**Solutions**:
1. Wait until first validation step (step 50 with current config)
2. Verify TensorBoard is pointing to correct directory
3. Refresh browser (Ctrl+F5)
4. Check validation is enabled in config: `validation.enabled: true`

### Cache Building Fails

**Symptoms**: Error during `--cache` step

**Solutions**:
1. Ensure dataset exists at specified path
2. Check you have ~1.4 GB free disk space for cache
3. Verify all images can be loaded: check for corrupted files
4. Try building cache in smaller batches by temporarily reducing dataset size

### Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'qflux'`

**Solutions**:
1. Always run from project root: `cd /home/ubuntu/sanjana-fs/qwen-image-finetune`
2. Check virtual environment is activated
3. Install requirements: `pip install -r requirements.txt`
4. Verify Python path includes src/: `echo $PYTHONPATH`

### Checkpoints Not Saving

**Symptoms**: Training completes but no checkpoint files

**Solutions**:
1. Check disk space: `df -h`
2. Verify output directory permissions
3. Ensure `checkpointing_steps` is set in config
4. Check logs for write errors

---

## 💡 Pro Tips

### Early Stopping
If validation images look perfect before training completes:
- Press `Ctrl+C` once (graceful shutdown)
- Wait for "saving last checkpoint" message
- Use the last checkpoint for inference
- No need to train longer!

### Comparing Checkpoints
To find the best checkpoint:
1. Check TensorBoard loss curves (lower is better)
2. Review validation images at different steps
3. Test multiple checkpoints on your specific use case
4. Consider checkpoints near local minima in loss curve

### Monitoring During Training
Key metrics to watch:
```
fit: 1234it [12:34, 3.45s/it, loss=0.123, smooth_loss=0.456, lr=1.2e-05, epoch=2, fps=0.58]
```
- `loss`: Current batch loss (fluctuates)
- `smooth_loss`: Exponential moving average (more stable indicator)
- `lr`: Current learning rate (should decrease over time with cosine scheduler)
- `fps`: Samples processed per second

### GPU Utilization
Monitor with `nvidia-smi`:
- **GPU Memory**: Should be ~30-35 GB on A100 40GB
- **GPU Utilization**: Should be 95-100% during training
- **Temperature**: Should stay below 80°C

### Backing Up Results
On rented cloud GPUs, always backup:
1. **Trained checkpoints** → HuggingFace Hub or cloud storage
2. **TensorBoard logs** → Git or separate backup
3. **Training config** → Git (already saved in outputs/)

---

## 📤 Uploading to HuggingFace Hub

After training, you can share your model on HuggingFace Hub for backup or sharing:

### Quick Upload

```python
from huggingface_hub import HfApi, create_repo

# Configuration
repo_name = "your-username/easy-circle-flux-lora"
checkpoint_path = "outputs/easy_circle_lora/easy_circle_flux_kontext/v0/checkpoint-5-20100"

# Create API client
api = HfApi()

# Create repository (optional)
create_repo(repo_name, repo_type="model", exist_ok=True, private=False)

# Upload the trained weights
api.upload_file(
    path_or_fileobj=f"{checkpoint_path}/pytorch_lora_weights.safetensors",
    path_in_repo="pytorch_lora_weights.safetensors",
    repo_id=repo_name,
    repo_type="model",
)

# Upload training config
api.upload_file(
    path_or_fileobj="outputs/easy_circle_lora/easy_circle_flux_kontext/v0/train_config.yaml",
    path_in_repo="train_config.yaml",
    repo_id=repo_name,
    repo_type="model",
)

print(f"✅ Model uploaded to: https://huggingface.co/{repo_name}")
```

### Authentication

Before uploading, authenticate with HuggingFace:

```bash
# Login with your HuggingFace token
huggingface-cli login
# Or use: hf auth login
```

Get your token from: https://huggingface.co/settings/tokens

### Upload Multiple Checkpoints

To upload multiple checkpoints for comparison:

```python
checkpoints = ["checkpoint-5-20100", "checkpoint-5-20000", "checkpoint-5-19900"]

for checkpoint_name in checkpoints:
    checkpoint_path = f"outputs/easy_circle_lora/easy_circle_flux_kontext/v0/{checkpoint_name}"
    
    # Upload weights to subfolder
    api.upload_file(
        path_or_fileobj=f"{checkpoint_path}/pytorch_lora_weights.safetensors",
        path_in_repo=f"{checkpoint_name}/pytorch_lora_weights.safetensors",
        repo_id=repo_name,
        repo_type="model",
    )
    
    # Upload state file
    api.upload_file(
        path_or_fileobj=f"{checkpoint_path}/state.json",
        path_in_repo=f"{checkpoint_name}/state.json",
        repo_id=repo_name,
        repo_type="model",
    )
```

### Create Model Card

Add a README.md to your HuggingFace repo with training details:

```python
readme_content = """---
license: apache-2.0
base_model: lrzjason/flux-kontext-nf4
tags:
- flux
- lora
- image-editing
---

# Easy Circle FLUX LoRA

LoRA fine-tuned on FLUX Kontext to add red dots to images.

## Training Details
- Base Model: lrzjason/flux-kontext-nf4
- Training Steps: 20,100
- Epochs: 5.7
- Batch Size: 2
- Learning Rate: 0.0001
- LoRA Rank: 16

## Usage
```python
from diffusers import FluxKontextPipeline

pipe = FluxKontextPipeline.from_pretrained("lrzjason/flux-kontext-nf4")
pipe.load_lora_weights("your-username/easy-circle-flux-lora")

# Generate
output = pipe(prompt="add red dots", image=input_image)
```
"""

api.upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id=repo_name,
    repo_type="model",
)
```

---

## 📚 Next Steps After Training

1. **Evaluate Results**
   - Compare checkpoints using validation metrics
   - Test on held-out test set (data/easy_circle/test/)
   - Visually inspect outputs at different training steps

2. **Backup Your Model**
   - Upload best checkpoint to HuggingFace Hub (see above)
   - Save TensorBoard logs for later analysis
   - Document training configuration and results

3. **Experiment Further**
   - Try different LoRA ranks (8, 16, 32)
   - Adjust learning rate (try 5e-5 or 2e-4)
   - Experiment with different schedulers (constant, linear, cosine)
   - Add more training data for better generalization

4. **Production Deployment**
   - Optimize inference speed (lower `num_inference_steps`)
   - Batch inference requests for efficiency
   - Set up model versioning and A/B testing

---

## 📊 Evaluation

After training, evaluate counting accuracy on the test set:

### Running the Evaluation

```bash
python scripts/easy_circle/eval_counting.py \
    --checkpoint outputs/easy_circle_lora/easy_circle_flux_kontext/v0/checkpoint-5-17500 \
    --num_samples 1500
```

**Options:**
- `--checkpoint`: Path to trained checkpoint directory (required)
- `--test_dataset`: Path to test dataset (default: data/easy_circle/test)
- `--num_samples`: Number of samples to evaluate (default: all 1,500)
- `--num_inference_steps`: Diffusion steps (default: 20)
- `--cfg_scale`: Guidance scale (default: 3.5)
- `--sam3_device`: Device for SAM3 model (default: cuda:0)
- `--sam3_score_threshold`: Minimum confidence score for detections (default: 0.3)

### What It Does

The evaluation script:
1. Loads the test dataset (1,500 held-out samples)
2. Initializes SAM3 (Segment Anything Model 3) for dot detection
3. Runs inference with both base model and fine-tuned model
4. Counts dots in generated images using SAM3 text-prompted segmentation
5. Calculates three metrics:
   - **Accuracy**: % of exact count matches
   - **Mean Absolute Error (MAE)**: Average counting error
   - **Median Absolute Error (MedAE)**: Median counting error
6. Compares base vs. fine-tuned performance

### Example Output

```
Easy Circle Counting Evaluation (Test Set: 1,500 images)

Metric                          Base Model    Fine-tuned    Δ
────────────────────────────────────────────────────────────
Accuracy (exact match)          12.3%         78.4%         +66.1 pp
Mean Absolute Error             5.2 dots      0.8 dots      -4.4 dots  
Median Absolute Error           4.0 dots      0.0 dots      -4.0 dots

Summary:
Fine-tuning improved exact counting accuracy from 12.3% to 78.4%, 
reducing average error from 5.2 to 0.8 dots.
```

### Quick Evaluation (100 samples)

For faster iteration during development:

```bash
python scripts/easy_circle/eval_counting.py \
    --checkpoint outputs/easy_circle_lora/easy_circle_flux_kontext/v0/checkpoint-5-17500 \
    --num_samples 100
```

### Dot Detection with SAM3

The evaluation uses [SAM3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3) for counting dots:

- **Text-prompted segmentation**: Uses "red dots" prompt to detect objects
- **No manual thresholding**: SAM3 understands semantic concepts without color space conversions
- **Confidence filtering**: Only counts detections above `sam3_score_threshold` (default: 0.3)
- **State-of-the-art accuracy**: SAM3 achieves high precision on instance segmentation tasks

**Why SAM3?**
- Superior to traditional OpenCV color thresholding
- Robust to lighting variations and overlapping dots
- Can distinguish individual dots even when clustered
- Leverages latest vision-language model capabilities

---

## 📖 Additional Resources

- **Project Repository**: https://github.com/sanjanachin/qwen-image-finetune
- **Base Model**: https://huggingface.co/lrzjason/flux-kontext-nf4
- **FLUX Documentation**: https://github.com/black-forest-labs/flux
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

---

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue or pull request in the main repository!
