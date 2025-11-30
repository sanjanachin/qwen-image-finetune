# 🎯 Easy Circle Fine-tuning - Quick Start

Your fine-tuning environment is fully configured and ready to go!

## ✅ Setup Complete

- ✅ Dataset: 10,000 samples (7,000 train / 1,500 val / 1,500 test)
- ✅ Config: `configs/easy_circle_flux_kontext.yaml`
- ✅ Model: FLUX Kontext with LoRA (BF16)
- ✅ Hardware: A100 40GB optimized
- ✅ Training script: `scripts/easy_circle/train_easy_circle.sh`

## 🚀 Start Training (Recommended Method)

### Option 1: Interactive Script (Easiest)

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/scripts/easy_circle
./train_easy_circle.sh
```

The script will guide you through:
1. Building the embedding cache (recommended: yes)
2. Starting training

### Option 2: Manual Commands

**Step 1: Build Cache** (10-15 minutes, gives 2-3x speedup)
```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/src
python -m qflux.main \
    --config ../configs/easy_circle_flux_kontext.yaml \
    --cache
```

**Step 2: Train** (2-3 hours for 3000 steps)
```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/src
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file ../accelerate_config_single_gpu.yaml \
    -m qflux.main \
    --config ../configs/easy_circle_flux_kontext.yaml
```

## 📊 Monitor Training

In a **separate terminal**, launch TensorBoard:

```bash
tensorboard --logdir=/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora
```

Then open **http://localhost:6006** to view:
- 📈 Loss curves in real-time
- 🖼️ Validation images every 100 steps
- 📉 Learning rate schedule

## 📁 Where Things Are Saved

```
outputs/easy_circle_lora/
├── easy_circle_flux_kontext/
│   ├── checkpoint-200/    # Step 200
│   ├── checkpoint-400/    # Step 400
│   ├── ...
│   ├── checkpoint-3000/   # Final checkpoint ← Use this!
│   └── cache/             # Embedding cache
└── events.out.tfevents.*  # TensorBoard logs
```

## 🧪 Test Your Trained Model

After training completes, test it:

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/scripts/easy_circle

python test_inference.py \
    --checkpoint ../../outputs/easy_circle_lora/checkpoint-3000 \
    --num_dots 10 \
    --output my_result.png
```

This will:
1. Load your trained LoRA weights
2. Generate a blank white image
3. Add 10 red dots to it
4. Save the result as `my_result.png`

## ⏱️ Expected Timeline

| Stage | Time | Notes |
|-------|------|-------|
| Cache Building | 10-15 min | One-time, optional but recommended |
| Training (3000 steps) | 2-3 hours | ~6 seconds per step |
| Validation | +5-10 min | Every 100 steps (automatic) |
| **Total** | **~2.5-3.5 hours** | With cache enabled |

## 🎨 Training Details

- **Task**: Learn to add red dots to blank white images
- **Model**: FLUX Kontext (memory-efficient diffusion model)
- **Method**: LoRA (Low-Rank Adaptation) - only trains 0.5% of parameters
- **Precision**: BF16 (best quality on A100)
- **Batch Size**: 2
- **Steps**: 3000 (~4.3 epochs)
- **Validation**: Every 100 steps with 4 samples
- **Checkpoints**: Every 200 steps

## 💡 Tips

### If Training is Interrupted
Edit `configs/easy_circle_flux_kontext.yaml` and add:
```yaml
resume: /home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora/checkpoint-XXXX
```
Then rerun the training command.

### If You Get Out of Memory
Reduce batch size in config:
```yaml
data:
  batch_size: 1  # Change from 2 to 1
```

### Saving to HuggingFace (Optional)
After training, you can upload your weights:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="outputs/easy_circle_lora/checkpoint-3000",
    repo_id="your-username/easy-circle-lora",
    repo_type="model"
)
```

## 📚 Documentation

- **Full Guide**: `scripts/easy_circle/README.md`
- **Config Reference**: `docs/guide/configuration.md`
- **Training Guide**: `docs/guide/training.md`
- **Inference Guide**: `docs/guide/inference.md`

## ❓ Troubleshooting

**Import Errors**: Always run from `src/` directory
**Cache Not Working**: Check `use_cache: true` in config
**No Validation Images**: Wait until step 100

---

## 🎯 Ready to Start?

Run this command to begin:

```bash
cd /home/ubuntu/sanjana-fs/qwen-image-finetune/scripts/easy_circle && ./train_easy_circle.sh
```

Good luck with your fine-tuning! 🚀

