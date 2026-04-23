#!/bin/bash
# Counting SFT: Fine-tune Qwen-Image-Edit with LoRA to obey object counts.
#
# Two phases:
#   1. Build embedding cache (VAE latents + Qwen2.5-VL prompt embeddings)
#   2. Train the DiT with LoRA using cached embeddings
#
# Usage:
#   bash scripts/counting/train_counting.sh              # full run (~3 epochs)
#   bash scripts/counting/train_counting.sh --smoke-test  # 10-step smoke test
#
# Prerequisites:
#   1. Download the dataset first:
#      python scripts/counting/download_data.py
#   2. Activate the conda environment:
#      conda activate qwen-image-edit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export QFLUX_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_FILE="$QFLUX_REPO_ROOT/configs/counting_qwen_image_edit.yaml"
ACCELERATE_CONFIG="$QFLUX_REPO_ROOT/accelerate_config_single_gpu.yaml"

SMOKE_TEST=false
if [[ "${1:-}" == "--smoke-test" ]]; then
    SMOKE_TEST=true
fi

echo "======================================"
echo "Counting SFT Fine-tuning"
echo "======================================"
echo "Model:   Qwen-Image-Edit"
echo "Task:    Obey object counts in image editing"
echo "Config:  $CONFIG_FILE"
echo "Repo:    $QFLUX_REPO_ROOT"
echo "======================================"
echo ""

# --- Validate prerequisites ---
if [[ ! -d "$QFLUX_REPO_ROOT/src" ]]; then
    echo "Error: Cannot find src/ directory at $QFLUX_REPO_ROOT"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config not found at $CONFIG_FILE"
    exit 1
fi

DATA_DIR="$QFLUX_REPO_ROOT/data/counting/train"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Training data not found at $DATA_DIR"
    echo "Run the download script first:"
    echo "  python scripts/counting/download_data.py"
    exit 1
fi

PARQUET_COUNT=$(find "$DATA_DIR" -name "*.parquet" 2>/dev/null | wc -l)
if (( PARQUET_COUNT == 0 )); then
    echo "Error: No .parquet files found in $DATA_DIR"
    echo "Run the download script first:"
    echo "  python scripts/counting/download_data.py"
    exit 1
fi
echo "Training data: $PARQUET_COUNT parquet file(s) in $DATA_DIR"
echo ""

cd "$QFLUX_REPO_ROOT/src"

# --- Phase 1 + 2 ---
if $SMOKE_TEST; then
    echo "Phase 2: SMOKE TEST (10 steps)..."
    echo "Creating temporary smoke-test config..."

    SMOKE_CONFIG="$QFLUX_REPO_ROOT/configs/_counting_smoke_test.yaml"
    cp "$CONFIG_FILE" "$SMOKE_CONFIG"
    # Use the small val split for both train and validation data to keep caching fast
    python -c "
import yaml
with open('$SMOKE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['train']['max_train_steps'] = 10
cfg['train']['checkpointing_steps'] = 5
cfg['validation']['steps'] = 5
val_path = '$QFLUX_REPO_ROOT/data/counting/val'
cfg['data']['init_args']['dataset_path'] = [val_path]
cfg['logging']['output_dir'] = '$QFLUX_REPO_ROOT/outputs/counting_lora_smoke'
cfg['cache']['cache_dir'] = '\${logging.output_dir}/cache'
with open('$SMOKE_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('Smoke test: 10 steps, checkpoint@5, validate@5, data=val (216 samples)')
"

    # Cache phase for smoke test
    echo "Building smoke-test cache (val split only, ~216 samples)..."
    python -m qflux.main --config "$SMOKE_CONFIG" --cache

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        -m qflux.main \
        --config "$SMOKE_CONFIG"

    rm -f "$SMOKE_CONFIG"
    echo ""
    echo "Smoke test complete! Check outputs at:"
    echo "  $QFLUX_REPO_ROOT/outputs/counting_lora_smoke/"
else
    # --- Full run: cache then train ---
    echo "Phase 1: Building embedding cache..."
    echo "This pre-computes VAE latents and VL prompt embeddings."
    python -m qflux.main --config "$CONFIG_FILE" --cache
    echo ""

    echo "Phase 2: Starting training..."
    echo "  Steps:        ~88,100 (10 epochs)"
    echo "  Batch size:   1 (effective 2 with grad accum)"
    echo "  Checkpoints:  every 500 steps"
    echo "  Validation:   every 250 steps"
    echo ""

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        -m qflux.main \
        --config "$CONFIG_FILE"

    echo ""
    echo "Training complete!"
fi

echo ""
echo "Outputs:     $QFLUX_REPO_ROOT/outputs/counting_lora/"
echo "TensorBoard: tensorboard --logdir=$QFLUX_REPO_ROOT/outputs/counting_lora/"
echo ""
echo "Next steps:"
echo "  1. Review training curves in TensorBoard"
echo "  2. Merge LoRA into base weights for GRPO:"
echo "     python scripts/counting/merge_lora.py \\"
echo "       --checkpoint outputs/counting_lora/<version>/checkpoint-<epoch>-<step> \\"
echo "       --output-dir outputs/counting_merged"
