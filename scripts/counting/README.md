# Counting SFT Pipeline

Fine-tune [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) with LoRA to obey object counts in image editing prompts. This is the SFT stage of a three-part pipeline: **dataset creation -> SFT -> RL (GRPO)**.

## Overview

| Component | Description |
|-----------|-------------|
| **Model** | Qwen-Image-Edit (DiT with Qwen2.5-VL text encoder) |
| **Method** | LoRA on attention layers (`to_k`, `to_q`, `to_v`, `to_out.0`) |
| **Dataset** | 17,620 train / 216 val / 216 test counting image-edit pairs (parquet) |
| **Training** | ~3 epochs, bf16 mixed precision, Adam8bit, cosine LR schedule |
| **Hardware** | Single GPU with ≥80 GB VRAM (e.g. H100 80GB SXM5) |

## Quick Start

### 1. Download the dataset

```bash
python scripts/counting/download_data.py
```

This downloads ~29GB of parquet files from S3. The training pipeline reads parquet natively — no conversion to individual image files is needed. See `--help` for options like `--splits train` to download a single split.

### 2. Run training

```bash
bash scripts/counting/train_counting.sh
```

The script handles both phases automatically:
1. **Embedding cache** — pre-computes VAE latents and Qwen2.5-VL prompt embeddings (reads images directly from parquet)
2. **LoRA training** — trains the DiT with cached embeddings

To run a quick 10-step smoke test first:

```bash
bash scripts/counting/train_counting.sh --smoke-test
```

### 3. Monitor training

```bash
tensorboard --logdir=outputs/counting_lora
```

TensorBoard shows:
- Training / smoothed loss
- Learning rate schedule
- Validation images every 250 steps

### 4. Merge LoRA for GRPO

After training, merge the LoRA adapter into the base model so the [counting-grpo](https://github.com/your-org/counting-grpo) RL pipeline can load it:

```bash
python scripts/counting/merge_lora.py \
    --checkpoint outputs/counting_lora/counting_qwen_image_edit/v0/checkpoint-2-26430 \
    --output-dir outputs/counting_merged
```

The merged model can then be loaded by the GRPO script via:

```python
QwenImageTransformer2DModel.from_pretrained("outputs/counting_merged", subfolder="transformer")
```

## Files

| File | Purpose |
|------|---------|
| `scripts/counting/download_data.py` | Download parquet data from S3 |
| `scripts/counting/train_counting.sh` | End-to-end training script (cache -> train) |
| `scripts/counting/merge_lora.py` | Merge LoRA weights into base model for downstream RL |
| `configs/counting_qwen_image_edit.yaml` | Training configuration (hyperparameters, paths, validation) |

## Training Configuration

Key hyperparameters (from `configs/counting_qwen_image_edit.yaml`):

```
LoRA rank:            16
Learning rate:        1e-4 (Adam8bit)
LR schedule:          cosine with 100-step warmup
Batch size:           1 (effective 2 with gradient accumulation)
Mixed precision:      bf16
Gradient checkpoint:  enabled
Max steps:            26,430 (~3 epochs)
Checkpoints:          every 500 steps (keep last 5)
Validation:           every 250 steps (4 samples)
```

These mirror the hyperparameters from the successful easy_circle proof-of-concept run.

## Dataset Format

The dataset is stored as parquet files on S3 (`s3://counting-dataset/count-dataset-splits/`). The training pipeline reads them directly using native parquet support in `ImageDataset`. Key columns:

| Column | Role |
|--------|------|
| `original_image` | Input image (control) |
| `edited_image` | Ground-truth edited image (target) |
| `prompt` | Editing instruction, e.g. "add 3 cats to the image" |
| `object_name` | Type of object added |
| `count_added` | Number of objects added |

The column mapping is configured in the YAML config via `parquet_column_map`.

## Output Structure

```
outputs/counting_lora/
├── cache/                              # Cached embeddings
│   ├── control_latents/
│   ├── image_latents/
│   ├── prompt_embeds/
│   └── metadata/
└── counting_qwen_image_edit/
    └── v0/
        ├── checkpoint-0-500/           # checkpoint-{epoch}-{step}
        │   ├── pytorch_lora_weights.safetensors
        │   └── state.json
        ├── checkpoint-0-1000/
        ├── ...
        ├── events.out.tfevents.*       # TensorBoard logs
        └── train_config.yaml           # Snapshot of config
```

## GPU Requirements

The Qwen-Image-Edit transformer is ~24 GB in bf16. With LoRA adapter weights, Adam8bit optimizer states, and activation memory during training, **a single GPU with ≥80 GB VRAM is required** (e.g. H100 80GB SXM5, A100 80GB). The model runs in full bf16 precision with no quantization, which produces the cleanest LoRA weights for the downstream RL stage.

A 40 GB GPU (e.g. A100 40GB) is **not** sufficient for full-precision training. If you only have 40 GB, you would need to use the pre-quantized `ovedrive/qwen-image-edit-4bit` model (NF4, ~6 GB transformer) by changing `pretrained_model_name_or_path` in the config. However, the default config targets 80 GB GPUs with full precision for best quality.

## Troubleshooting

**OOM**: The config uses `batch_size: 1`, `gradient_checkpointing: true`, and `low_memory: true`. If you still OOM, try reducing `gradient_accumulation_steps` to 1, reducing LoRA rank to 8, or reducing `target_size` to `[384, 384]`.

**Slow training**: Make sure the embedding cache was built (Phase 1). Training without cache requires loading the full VAE and text encoder alongside the DiT.

**Import errors**: Training runs from `src/` via `python -m qflux.main`. Make sure the conda environment is activated and all dependencies are installed.
