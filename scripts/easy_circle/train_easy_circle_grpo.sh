#!/bin/bash
# Easy Circle GRPO Training Script
# This script trains FLUX Kontext using GRPO to improve counting accuracy

set -e  # Exit on error

# Configuration
CONFIG_FILE="../configs/easy_circle_grpo.yaml"
ACCELERATE_CONFIG="../accelerate_config_single_gpu.yaml"

echo "======================================"
echo "Easy Circle GRPO Training"
echo "======================================"
echo "Model: FLUX Kontext"
echo "Method: GRPO (Group Relative Policy Optimization)"
echo "GPU: A100 40GB"
echo "Task: Improve red dot counting accuracy"
echo "======================================"
echo ""
echo "GRPO Training Parameters:"
echo "  - K samples per prompt: 4"
echo "  - Prompts per step: 2"
echo "  - KL penalty beta: 0.01"
echo "  - Exact match bonus: 2.0"
echo "  - Learning rate: 1e-5"
echo ""

# Check if we're in the right directory
if [ ! -d "../../src" ]; then
    echo "Error: Please run this script from the scripts/easy_circle/ directory"
    exit 1
fi

# Navigate to src directory (required for imports)
cd ../../src

# Note: GRPO doesn't use cache - it needs raw images for reward computation
echo "Note: GRPO training doesn't use embedding cache."
echo "      It needs raw images to compute SAM3-based counting rewards."
echo ""

# Step 1: Start GRPO training
echo "Starting GRPO training..."
echo "TEST RUN: Training will run for 2000 steps"
echo "Validation samples will be generated every 50 steps"
echo "Checkpoints saved every 200 steps"
echo ""

read -p "Start GRPO training? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        -m qflux.main \
        --config $CONFIG_FILE
    
    echo ""
    echo "✅ GRPO Training complete!"
    echo ""
    echo "📁 Outputs saved to: /home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_grpo"
    echo "📊 View training logs: tensorboard --logdir=/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_grpo"
    echo ""
    echo "🎯 To evaluate the trained model, run:"
    echo "   python scripts/easy_circle/eval_counting.py \\"
    echo "       --checkpoint outputs/easy_circle_grpo/easy_circle_grpo/v0/checkpoint-X-XXXX \\"
    echo "       --balanced_test"
else
    echo "Training cancelled."
fi

