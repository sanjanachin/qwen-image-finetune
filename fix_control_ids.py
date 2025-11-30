#!/usr/bin/env python3
"""Quick script to generate the missing control_ids cache file."""
import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, '/home/ubuntu/sanjana-fs/qwen-image-finetune/src')

from qflux.utils.options import parse_args

# Parse config
sys.argv = ['fix_control_ids.py', '--config', '/home/ubuntu/sanjana-fs/qwen-image-finetune/configs/easy_circle_flux_kontext.yaml']
config = parse_args()

print("Loading FluxKontext model...")
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

trainer = FluxKontextLoraTrainer(config)
trainer.load_model()
trainer.setup_model_device_train_mode(stage="cache", cache=True)

print("Loading one control image...")
# Load the control image directly
control_path = "/home/ubuntu/sanjana-fs/qwen-image-finetune/data/easy_circle/train/control_images/sample_00000.png"
control_img = Image.open(control_path).convert("RGB")
control_img = control_img.resize((512, 512))
control_array = np.array(control_img).astype(np.float32) / 255.0
control_tensor = torch.from_numpy(control_array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]

print(f"Control tensor shape: {control_tensor.shape}")

# Move to device and encode
device = torch.device("cuda:0")
control_tensor = control_tensor.to(device).to(trainer.weight_dtype)
with torch.no_grad():
    # Encode control image
    control_latents = trainer.vae.encode(control_tensor).latent_dist.sample()
    control_latents = (control_latents - trainer.vae.config.shift_factor) * trainer.vae.config.scaling_factor
    
    # Generate control_ids (positional encodings)
    height, width = control_latents.shape[2], control_latents.shape[3]
    control_ids = trainer._prepare_latent_image_ids(batch_size=1, height=height, width=width, device=device, dtype=trainer.weight_dtype)

print(f"Generated control_ids shape: {control_ids.shape}")

# Save to cache
cache_path = Path("/home/ubuntu/sanjana-fs/qwen-image-finetune/outputs/easy_circle_lora/cache/control_ids/44d7c9580f4f153ead838b6beadb510c.pt")
cache_path.parent.mkdir(parents=True, exist_ok=True)

control_ids_cpu = control_ids.detach().cpu().to(torch.float16)
torch.save(control_ids_cpu, cache_path)

print(f"✅ Created missing control_ids file: {cache_path}")
print(f"✅ File size: {cache_path.stat().st_size} bytes")

# Verify it loads
test_load = torch.load(cache_path, map_location="cpu", weights_only=False)
print(f"✅ Verified file loads correctly, shape: {test_load.shape}")
print("\n🎉 Cache is now complete! Restart training.")

