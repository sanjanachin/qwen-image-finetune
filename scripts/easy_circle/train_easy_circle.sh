#!/bin/bash
# Easy Circle Fine-tuning Training Script
# This script trains FLUX Kontext to add red dots to blank images

set -e  # Exit on error

# Configuration
CONFIG_FILE="../configs/easy_circle_flux_kontext.yaml"
ACCELERATE_CONFIG="../accelerate_config_single_gpu.yaml"

echo "======================================"
echo "Easy Circle Fine-tuning"
echo "======================================"
echo "Model: FLUX Kontext"
echo "GPU: A100 40GB"
echo "Task: Add red dots to blank images"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -d "../../src" ]; then
    echo "Error: Please run this script from the scripts/easy_circle/ directory"
    exit 1
fi

# Navigate to src directory (required for imports)
cd ../../src

# Step 1: Check and build cache if needed
CACHE_DIR="/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora/cache"
if [ -d "$CACHE_DIR/metadata" ]; then
    CACHED_COUNT=$(find "$CACHE_DIR/metadata" -name "*.json" | wc -l)
    EXPECTED_COUNT=7000
    
    if [ "$CACHED_COUNT" -eq "$EXPECTED_COUNT" ]; then
        echo "Step 1: ✅ Cache is complete!"
        echo "Found $CACHED_COUNT/$EXPECTED_COUNT cached samples"
        echo "Skipping cache building..."
        echo ""
    else
        echo "Step 1: ⚠️  Cache is incomplete!"
        echo "Found $CACHED_COUNT/$EXPECTED_COUNT cached samples (missing $(($EXPECTED_COUNT - $CACHED_COUNT)))"
        echo "Completing cache... This will only process the missing samples."
        python -m qflux.main --config $CONFIG_FILE --cache
        echo "✅ Cache completed!"
        echo ""
    fi
else
    echo "Step 1: Building embedding cache..."
    echo "This precomputes VAE and text encoder outputs to speed up training."
    read -p "Build cache? (y/n, recommended: y): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Building cache... This may take 10-15 minutes."
        python -m qflux.main --config $CONFIG_FILE --cache
        echo "✅ Cache built successfully!"
        echo ""
    else
        echo "⚠️  Skipping cache. Training will be slower."
        echo ""
    fi
fi

# Step 2: Start training
echo "Step 2: Starting training..."
echo "TEST RUN: Training will run for 300 steps (~30 minutes)"
echo "Validation samples will be generated every 50 steps"
echo "Checkpoints saved every 100 steps"
echo ""
read -p "Start training? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        -m qflux.main \
        --config $CONFIG_FILE
    
    echo ""
    echo "✅ Training complete!"
    echo ""
    echo "📁 Outputs saved to: /home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora"
    echo "📊 View training logs: tensorboard --logdir=/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora"
    echo ""
    echo "🎯 Your trained LoRA weights are in the checkpoints directory!"
else
    echo "Training cancelled."
fi

