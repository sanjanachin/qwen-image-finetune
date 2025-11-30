#!/usr/bin/env python3
"""
Upload trained LoRA checkpoints to HuggingFace Hub.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_NAME = "sanjanachintalapati/easy-circle-flux-lora"  # Updated to match HF username
CHECKPOINT_DIR = "outputs/easy_circle_lora/easy_circle_flux_kontext/v0"

# Checkpoints to upload (most recent ones)
CHECKPOINTS_TO_UPLOAD = [
    "checkpoint-5-20100",  # Latest
    "checkpoint-5-20000",
    "checkpoint-5-19900",
]

def upload_checkpoints():
    """Upload selected checkpoints to HuggingFace Hub."""
    api = HfApi()
    
    # Create repository (will skip if already exists)
    print(f"Creating/accessing repository: {REPO_NAME}")
    try:
        create_repo(REPO_NAME, repo_type="model", exist_ok=True, private=False)
        print(f"✓ Repository ready: https://huggingface.co/{REPO_NAME}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload each checkpoint
    for checkpoint_name in CHECKPOINTS_TO_UPLOAD:
        checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"⚠ Skipping {checkpoint_name} - not found")
            continue
        
        print(f"\nUploading {checkpoint_name}...")
        
        # Upload the LoRA weights
        weights_file = checkpoint_path / "pytorch_lora_weights.safetensors"
        state_file = checkpoint_path / "state.json"
        
        if weights_file.exists():
            api.upload_file(
                path_or_fileobj=str(weights_file),
                path_in_repo=f"{checkpoint_name}/pytorch_lora_weights.safetensors",
                repo_id=REPO_NAME,
                repo_type="model",
            )
            print(f"  ✓ Uploaded weights ({weights_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        if state_file.exists():
            api.upload_file(
                path_or_fileobj=str(state_file),
                path_in_repo=f"{checkpoint_name}/state.json",
                repo_id=REPO_NAME,
                repo_type="model",
            )
            print(f"  ✓ Uploaded state")
    
    # Upload training config
    config_file = Path(CHECKPOINT_DIR) / "train_config.yaml"
    if config_file.exists():
        print(f"\nUploading training config...")
        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo="train_config.yaml",
            repo_id=REPO_NAME,
            repo_type="model",
        )
        print(f"  ✓ Uploaded training config")
    
    # Create a README
    readme_content = f"""---
license: apache-2.0
base_model: lrzjason/flux-kontext-nf4
tags:
- flux
- lora
- image-editing
- synthetic-data
---

# Easy Circle FLUX LoRA

LoRA fine-tuned on FLUX Kontext model to add red dots to images.

## Training Details

- **Base Model**: lrzjason/flux-kontext-nf4
- **Task**: Add red dots on specified objects in images
- **Dataset**: 7,000 synthetic image pairs (blank → blank with red dots)
- **Training Steps**: 20,100 steps over 5.7 epochs
- **Training Time**: ~20 hours on A100 GPU
- **LoRA Rank**: 16
- **Learning Rate**: 0.0001 with cosine scheduler
- **Batch Size**: 2
- **Image Resolution**: 512x512

## Available Checkpoints

- `checkpoint-5-20100/` - Final checkpoint (step 20,100)
- `checkpoint-5-20000/` - Step 20,000
- `checkpoint-5-19900/` - Step 19,900

## Usage

```python
from diffusers import FluxKontextPipeline
import torch

# Load base model
pipe = FluxKontextPipeline.from_pretrained("lrzjason/flux-kontext-nf4")

# Load LoRA weights
pipe.load_lora_weights("sanjanachintalapati/easy-circle-flux-lora", 
                       subfolder="checkpoint-5-20100")

# Generate
prompt = "Add red dots"
control_image = ... # Your input image
output = pipe(prompt=prompt, image=control_image)
```

## Training Code

Full training code and configuration available at: https://github.com/sanjanachin/qwen-image-finetune
"""
    
    print(f"\nCreating README...")
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=REPO_NAME,
        repo_type="model",
    )
    print(f"  ✓ Created README")
    
    print(f"\n{'='*60}")
    print(f"✅ Upload complete!")
    print(f"🔗 View at: https://huggingface.co/{REPO_NAME}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check if logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to HuggingFace!")
        print("\nPlease run:")
        print("  huggingface-cli login")
        print("\nOr set HF_TOKEN environment variable")
        exit(1)
    
    upload_checkpoints()

