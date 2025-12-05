"""
GRPO (Group Relative Policy Optimization) Trainer for Flux Kontext

This trainer implements GRPO for fine-tuning Flux Kontext to improve counting accuracy.
Instead of using pixel-level MSE loss, it uses reinforcement learning with a 
non-differentiable reward function (SAM3-based dot counting).

Key differences from standard SFT:
1. Generate K samples per prompt
2. Compute rewards via SAM3 counting
3. Compute group-relative advantages
4. Update model using policy gradient loss
5. Add KL penalty to prevent divergence from reference model
"""

import gc
import logging
import os
import re
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import Config

logger = logging.getLogger(__name__)

# SAM3 path - will be set from config
SAM3_PATH = "/home/ubuntu/sanjana-fs/sam3"


class GRPOFluxKontextTrainer(FluxKontextLoraTrainer):
    """
    GRPO Trainer for Flux Kontext.
    
    Extends FluxKontextLoraTrainer to use GRPO instead of standard SFT.
    Uses SAM3 for non-differentiable reward computation.
    """

    def __init__(self, config: Config):
        """Initialize GRPO trainer with configuration."""
        super().__init__(config)
        
        # GRPO-specific configuration
        grpo_config = config.grpo
        if grpo_config is None:
            # Default GRPO settings
            self.k_samples = 4
            self.prompts_per_step = 2
            self.kl_beta = 0.01
            self.exact_match_bonus = 2.0
            self.sam3_path = SAM3_PATH
            self.sam3_confidence_threshold = 0.3
            self.use_sam3_api = False
            self.sam3_api_max_masks = 30
        else:
            self.k_samples = grpo_config.k_samples
            self.prompts_per_step = grpo_config.prompts_per_step
            self.kl_beta = grpo_config.kl_beta
            self.exact_match_bonus = grpo_config.exact_match_bonus
            self.sam3_path = grpo_config.sam3_path
            self.sam3_confidence_threshold = grpo_config.sam3_confidence_threshold
            self.use_sam3_api = grpo_config.use_sam3_api
            self.sam3_api_max_masks = grpo_config.sam3_api_max_masks
        
        # SAM3 processor (initialized later, only if not using API)
        self.sam3_processor = None
        self.sam3_device = "cuda:0"
        
        # Reference model for KL penalty (frozen copy)
        self.reference_dit = None
        
        # GRPO metrics tracking
        self.reward_history = []
        self.kl_history = []
        self.accuracy_history = []
        
        logger.info(f"GRPO Config: k_samples={self.k_samples}, prompts_per_step={self.prompts_per_step}")
        logger.info(f"GRPO Config: kl_beta={self.kl_beta}, exact_match_bonus={self.exact_match_bonus}")

    def setup_model_device_train_mode(self, stage="fit", cache=False):
        """
        Override to keep VAE decoder on GPU for GRPO.
        
        GRPO needs to decode generated latents to images for reward computation,
        so we cannot move the VAE decoder to CPU like standard training does.
        """
        if stage == "fit" and not cache:
            # Move all models to GPU
            self.vae.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
            self.text_encoder_2.to(self.accelerator.device)
            self.dit.to(self.accelerator.device)
            
            # IMPORTANT: Keep VAE decoder on GPU for GRPO (unlike standard training)
            # Standard training moves it to CPU to save memory, but GRPO needs it
            # to decode generated images for reward computation
            logger.info("GRPO: Keeping VAE decoder on GPU for reward computation")
            
            # Set requires_grad and eval/train modes
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.text_encoder_2.requires_grad_(False).eval()
            self.dit.requires_grad_(False)
            self.dit.train()
            
            # Enable gradients only for LoRA parameters
            for name, param in self.dit.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Use parent implementation for other stages
            super().setup_model_device_train_mode(stage, cache)

    def initialize_sam3(self):
        """Initialize SAM3 model for dot detection."""
        if self.sam3_processor is not None:
            return
            
        logger.info(f"Initializing SAM3 model on {self.sam3_device}...")
        
        # Add SAM3 to path
        sam3_path = self.sam3_path
        if os.path.exists(sam3_path):
            sys.path.insert(0, sam3_path)
        else:
            raise FileNotFoundError(
                f"SAM3 repository not found at {sam3_path}. "
                "Please clone it from https://github.com/facebookresearch/sam3"
            )
        
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Build model with BPE vocabulary
        bpe_path = os.path.join(sam3_path, "assets/bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE vocabulary file not found at {bpe_path}")
        
        model = build_sam3_image_model(bpe_path=bpe_path)
        model = model.to(self.sam3_device)
        
        self.sam3_processor = Sam3Processor(
            model, 
            confidence_threshold=self.sam3_confidence_threshold
        )
        logger.info("✅ SAM3 model initialized")

    def count_dots_in_image(self, image: Image.Image) -> int:
        """
        Count red dots in an image using SAM3.
        
        Uses either local SAM3 model or fal.ai API based on config.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Number of detected red dots
        """
        if self.use_sam3_api:
            return self._count_dots_via_api(image)
        else:
            return self._count_dots_local(image)

    def _count_dots_via_api(self, image: Image.Image) -> int:
        """
        Count red dots using SAM3 via fal.ai API.
        
        Requires FAL_KEY environment variable to be set.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Number of detected red dots
        """
        import tempfile
        import fal_client
        
        try:
            # Save image to temporary file for upload
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Upload image to fal.ai
            image_url = fal_client.upload_file(tmp_path)
            
            # Call SAM3 API
            result = fal_client.subscribe(
                "fal-ai/sam-3/image-rle",
                arguments={
                    "image_url": image_url,
                    "prompt": "red dot",
                    "return_multiple_masks": True,
                    "max_masks": self.sam3_api_max_masks,
                    "include_scores": False,
                    "include_boxes": False,
                },
                with_logs=False,
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Count masks from the result
            rle = result.get("rle")
            if rle is None:
                return 0
            elif isinstance(rle, list):
                return len(rle)
            else:
                # Single mask returned as string
                return 1
                
        except Exception as e:
            logger.warning(f"Error in SAM3 API call: {e}")
            return 0

    def _count_dots_local(self, image: Image.Image) -> int:
        """
        Count red dots using local SAM3 model.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Number of detected red dots
        """
        if self.sam3_processor is None:
            self.initialize_sam3()
        
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = self.sam3_processor.set_image(image)
                self.sam3_processor.reset_all_prompts(inference_state)
                inference_state = self.sam3_processor.set_text_prompt(
                    state=inference_state, 
                    prompt="red dot"
                )
                
                # Count detected objects
                if hasattr(inference_state, 'masks') and inference_state.masks is not None:
                    count = len(inference_state.masks)
                elif isinstance(inference_state, dict) and 'masks' in inference_state:
                    if inference_state['masks'] is not None:
                        count = len(inference_state['masks'])
                    else:
                        count = 0
                elif hasattr(inference_state, 'obj_ids'):
                    count = len(inference_state.obj_ids)
                else:
                    count = 0
                
                return count
                
        except Exception as e:
            logger.warning(f"Error in count_dots_in_image: {e}")
            return 0

    def extract_count_from_prompt(self, prompt: str) -> int:
        """Extract requested count from prompt like 'add 9 red dots to the image'."""
        match = re.search(r'add (\d+) red dots', prompt.lower())
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not parse count from prompt: {prompt}")

    def compute_reward(self, expected_count: int, actual_count: int) -> float:
        """
        Compute reward for a generated sample.
        
        Reward = -|expected - actual| + bonus × (exact_match)
        
        Args:
            expected_count: Number of dots requested in prompt
            actual_count: Number of dots detected by SAM3
            
        Returns:
            Reward value (higher is better)
        """
        error = abs(expected_count - actual_count)
        exact_match = 1.0 if error == 0 else 0.0
        reward = -error + self.exact_match_bonus * exact_match
        return reward

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """
        Compute GRPO-style advantages (group-relative normalization).
        
        advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
        
        Args:
            rewards: List of rewards for K samples
            
        Returns:
            List of advantages (same length as rewards)
        """
        rewards_array = np.array(rewards)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array) + 1e-8  # Avoid division by zero
        advantages = (rewards_array - mean_reward) / std_reward
        return advantages.tolist()

    def sample_with_gradients(
        self,
        control_image: torch.Tensor,
        prompt: str,
        num_inference_steps: int = 20,
    ) -> tuple[Image.Image, list[torch.Tensor]]:
        """
        Generate an image while tracking gradients through the diffusion process.
        
        This is the key method for policy gradient - we need to be able to
        backpropagate through the sampling process.
        
        Args:
            control_image: Control image tensor [1, C, H, W] normalized to [-1, 1]
            prompt: Text prompt
            num_inference_steps: Number of diffusion steps
            
        Returns:
            Tuple of (generated PIL image, list of noise predictions for each step)
        """
        device = next(self.dit.parameters()).device
        batch_size = 1
        
        # Prepare embeddings (reuse existing method)
        batch_data = {
            "control": control_image,
            "prompt": [prompt],
            "n_controls": 0,
        }
        
        # Normalize control if needed
        if control_image.max() > 1.0:
            control_image = control_image / 255.0
        batch_data["control"] = self.normalize_image(batch_data["control"])
        
        # Encode prompt
        pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_prompt(
            prompt=[prompt],
            prompt_2=None,
            max_sequence_length=self.max_sequence_length,
        )
        
        # Prepare control latents
        control = batch_data["control"].to(device)
        height, width = control.shape[2:]
        _, control_latents, _, control_ids = self.prepare_latents(
            control, batch_size, 16, height, width, self.weight_dtype
        )
        control_ids[..., 0] = 1
        
        # Create noise latents
        latents, latent_ids = self.create_sampling_latents(
            height, width, batch_size, 16, device, self.weight_dtype
        )
        latent_ids = torch.cat([latent_ids, control_ids], dim=0)
        image_seq_len = latents.shape[1]
        
        # Prepare timesteps
        timesteps, num_inference_steps = self.prepare_predict_timesteps(
            num_inference_steps, image_seq_len, scheduler=self.sampling_scheduler
        )
        
        # Guidance
        guidance = torch.ones((batch_size,), device=device, dtype=self.weight_dtype)
        
        # Move tensors to device
        pooled_prompt_embeds = pooled_prompt_embeds.to(device).to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(device).to(self.weight_dtype)
        text_ids = text_ids.to(device).to(self.weight_dtype)
        control_latents = control_latents.to(device).to(self.weight_dtype)
        
        self.sampling_scheduler.set_begin_index(0)
        
        # Store noise predictions for policy gradient
        noise_predictions = []
        
        # Sampling loop WITH gradients
        for t in timesteps:
            latent_model_input = torch.cat([latents, control_latents], dim=1)
            timestep = t.expand(batch_size).to(device).to(self.weight_dtype)
            
            # Forward pass - WITH gradients
            noise_pred = self.dit(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            
            noise_pred = noise_pred[:, :image_seq_len]
            noise_predictions.append(noise_pred)
            
            # Scheduler step (no gradients needed here)
            with torch.no_grad():
                latents = self.sampling_scheduler.step(
                    noise_pred.detach(), t, latents, return_dict=False
                )[0]
        
        # Decode to image
        with torch.no_grad():
            image_tensor = self.decode_vae_latent(latents.detach(), height, width)
            # Convert to PIL
            img_np = image_tensor[0].detach().permute(1, 2, 0).float().cpu().numpy()
            img_np = (img_np * 255).round().astype("uint8")
            pil_image = Image.fromarray(img_np)
        
        return pil_image, noise_predictions

    def compute_kl_penalty(
        self,
        current_noise_preds: list[torch.Tensor],
        reference_noise_preds: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty between current and reference model predictions.
        
        Approximates KL using MSE between noise predictions as a proxy.
        
        Args:
            current_noise_preds: Noise predictions from current model
            reference_noise_preds: Noise predictions from reference model
            
        Returns:
            KL penalty (scalar tensor)
        """
        kl_sum = 0.0
        for curr, ref in zip(current_noise_preds, reference_noise_preds):
            # Use MSE as proxy for KL
            kl_sum = kl_sum + torch.mean((curr - ref.detach()) ** 2)
        return kl_sum / len(current_noise_preds)

    def grpo_training_step(
        self,
        control_images: list[torch.Tensor],
        prompts: list[str],
        num_inference_steps: int = 20,
    ) -> dict[str, Any]:
        """
        Perform one GRPO training step.
        
        For each prompt:
        1. Generate K samples
        2. Compute rewards via SAM3
        3. Compute advantages
        4. Compute policy gradient loss
        
        Args:
            control_images: List of control image tensors
            prompts: List of text prompts
            num_inference_steps: Number of diffusion steps
            
        Returns:
            Dictionary with loss, rewards, advantages, etc.
        """
        device = next(self.dit.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        all_rewards = []
        all_advantages = []
        exact_matches = 0
        total_samples = 0
        
        for control_img, prompt in zip(control_images, prompts):
            expected_count = self.extract_count_from_prompt(prompt)
            
            # Generate K samples
            sample_rewards = []
            sample_noise_preds = []
            
            for k in range(self.k_samples):
                # Generate sample with gradients
                pil_image, noise_preds = self.sample_with_gradients(
                    control_img, prompt, num_inference_steps
                )
                
                # Count dots (no gradients - black box)
                with torch.no_grad():
                    actual_count = self.count_dots_in_image(pil_image)
                
                # Compute reward
                reward = self.compute_reward(expected_count, actual_count)
                sample_rewards.append(reward)
                sample_noise_preds.append(noise_preds)
                
                # Track accuracy
                if actual_count == expected_count:
                    exact_matches += 1
                total_samples += 1
            
            # Compute advantages
            advantages = self.compute_advantages(sample_rewards)
            
            all_rewards.extend(sample_rewards)
            all_advantages.extend(advantages)
            
            # Compute policy gradient loss for this prompt's samples
            for k in range(self.k_samples):
                advantage = advantages[k]
                noise_preds = sample_noise_preds[k]
                
                # Policy gradient: -advantage × log_prob
                # We approximate log_prob using negative MSE of noise predictions
                # Higher MSE = lower probability, so we use -MSE as log_prob proxy
                log_prob_proxy = torch.tensor(0.0, device=device)
                for noise_pred in noise_preds:
                    # Sum of squared predictions (proxy for log probability)
                    log_prob_proxy = log_prob_proxy - torch.mean(noise_pred ** 2)
                
                # Policy gradient loss component
                pg_loss = -advantage * log_prob_proxy / len(noise_preds)
                total_loss = total_loss + pg_loss
        
        # Normalize by number of samples
        num_samples = len(control_images) * self.k_samples
        total_loss = total_loss / num_samples
        
        # Compute metrics
        accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
        mean_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        return {
            "loss": total_loss,
            "mean_reward": mean_reward,
            "accuracy": accuracy,
            "rewards": all_rewards,
            "advantages": all_advantages,
        }

    def train_epoch_grpo(self, epoch: int, train_dataloader):
        """
        GRPO training epoch.
        
        Overrides standard training to use GRPO algorithm.
        """
        # Ensure SAM3 is initialized
        self.initialize_sam3()
        
        for batch_idx, batch in enumerate(train_dataloader):
            if self.training_interrupted:
                logger.info("Training interrupted, saving checkpoint...")
                self.save_checkpoint(epoch, self.global_step, is_last=True)
                return
            
            if self.global_step >= self.config.train.max_train_steps:
                logger.info(f"Reached max_train_steps ({self.config.train.max_train_steps})")
                return
            
            # Extract control images and prompts from batch
            # Batch structure: control [B, C, H, W], prompt [B]
            control_images = []
            prompts = []
            
            # Handle cached vs non-cached data
            if "control" in batch:
                controls = batch["control"]
            elif "control_latents" in batch:
                # Skip if only cached latents - we need raw images for GRPO
                logger.warning("GRPO requires raw images, not cached latents. Skipping batch.")
                continue
            else:
                continue
            
            batch_prompts = batch.get("prompt", [])
            
            # Limit to prompts_per_step
            num_prompts = min(len(batch_prompts), self.prompts_per_step)
            
            for i in range(num_prompts):
                control_images.append(controls[i:i+1])  # Keep batch dim
                prompts.append(batch_prompts[i] if isinstance(batch_prompts[i], str) else batch_prompts[i][0])
            
            if not prompts:
                continue
            
            # GRPO training step
            with self.accelerator.accumulate(self.dit):
                result = self.grpo_training_step(
                    control_images=control_images,
                    prompts=prompts,
                    num_inference_steps=20,
                )
                
                loss = result["loss"]
                
                # Backward pass
                self.accelerator.backward(loss)
                self.clip_gradients()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress and logging
            if self.accelerator.sync_gradients:
                self.train_loss = loss.item()
                self.running_loss = 0.9 * self.running_loss + 0.1 * self.train_loss
                
                # Log GRPO-specific metrics
                logs = {
                    "loss": self.train_loss,
                    "smooth_loss": self.running_loss,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "fps": self.fps_logger.total_fps(),
                    "reward_mean": result["mean_reward"],
                    "counting_accuracy": result["accuracy"],
                }
                
                self.update_progressbar(logs)
                self.save_checkpoint(epoch, self.global_step)
                
                # Validation
                if self.should_run_validation(self.global_step):
                    self.fps_logger.pause()
                    self.run_validation()
                    self.fps_logger.resume()

    def fit(self, train_dataloader):
        """
        Main training loop for GRPO.
        
        Similar to base fit() but uses GRPO training epochs.
        """
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        
        self.setup_signal_handlers()
        self.setup_accelerator()
        self.load_model()
        
        if self.config.resume is not None:
            import glob
            model_files = glob.glob(os.path.join(self.config.resume, "*.safetensors"))
            if len(model_files) > 0:
                self.config.model.lora.pretrained_weight = model_files[0]
            else:
                from qflux.trainer.constants import LORA_FILE_BASE_NAME
                self.config.model.lora.pretrained_weight = os.path.join(
                    self.config.resume, LORA_FILE_BASE_NAME
                )
            logging.info(f"Loaded checkpoint from {self.config.model.lora.pretrained_weight}")
        
        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit, self.config.predict.devices.dit
            )
        
        self.dit.to("cpu")
        self.setup_validation(train_dataloader.dataset)
        self.accelerator.wait_for_everyone()
        
        # Load pretrained LoRA if specified
        self.__class__.load_pretrain_lora_model(
            self.dit, self.config, self.adapter_name
        )
        
        # Setup for training
        self.setup_model_device_train_mode(stage="fit", cache=False)  # GRPO doesn't use cache
        self.configure_optimizers()
        self.setup_criterion()
        
        # Initialize SAM3
        logger.info("Initializing SAM3 for GRPO reward computation...")
        self.initialize_sam3()
        
        train_dataloader = self.accelerator_prepare(train_dataloader)
        
        logging.info("***** Running GRPO Training *****")
        logging.info(f"K samples per prompt: {self.k_samples}")
        logging.info(f"Prompts per step: {self.prompts_per_step}")
        logging.info(f"KL beta: {self.kl_beta}")
        logging.info(f"Exact match bonus: {self.exact_match_bonus}")
        
        from qflux.utils.model_summary import print_model_summary_table
        model_summary_info_dict = print_model_summary_table(self.dit)
        self.logger_manager.log_table(
            "model_summary",
            rows=model_summary_info_dict["rows"],
            columns=model_summary_info_dict["columns"],
            step=self.global_step,
        )
        
        self.fps_logger.start()
        self.save_train_config()
        self.setup_progressbar()
        
        current_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.num_epochs):
            current_epoch = epoch
            self.train_epoch_grpo(epoch, train_dataloader)
            if self.training_interrupted:
                break
        
        # Save final checkpoint
        self.save_checkpoint(current_epoch, self.global_step, is_last=True)
        
        logging.info(f"GRPO Training complete. Final step: {self.global_step}")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


# Additional GRPO configuration class for Pydantic config
class GRPOConfig:
    """GRPO-specific configuration (to be added to main Config)."""
    k_samples: int = 4
    prompts_per_step: int = 2
    kl_beta: float = 0.01
    exact_match_bonus: float = 2.0
    sam3_path: str = "/home/ubuntu/sanjana-fs/sam3"
    sam3_confidence_threshold: float = 0.3

