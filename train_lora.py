"""
Main training script for LoRA fine-tuning on Madhubani art
Supports LoRA, DoRA, and standard configurations
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb

from dataset import MadhubaniDataset, collate_fn
from config import LoRAConfig, DoRAConfig, get_config

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA/DoRA")
    parser.add_argument(
        "--config_name",
        type=str,
        default="lora_r32",
        help="Name of preset config (lora_r32, dora_r64, etc.)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Madhubani dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Use only a subset of the dataset (10, 20, or 50)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="madhubani-lora",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    
    args = parser.parse_args()
    return args


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR
    snr = (alpha / sigma) ** 2
    return snr


def main():
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config_name)
    
    # Override config with command line args
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Setup accelerator
    logging_dir = Path(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    if config.seed is not None:
        set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.config_name}_subset{args.subset_size}" if args.subset_size else args.config_name,
            config=vars(config)
        )
    
    # Load models
    logger.info(f"Loading models from {config.model_name}")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model_name,
        subfolder="tokenizer",
        revision=config.revision,
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        config.model_name,
        subfolder="text_encoder",
        revision=config.revision,
    )
    
    vae = AutoencoderKL.from_pretrained(
        config.model_name,
        subfolder="vae",
        revision=config.revision,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        config.model_name,
        subfolder="unet",
        revision=config.revision,
    )
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_dora=isinstance(config, DoRAConfig) and config.use_dora,
    )
    
    # Add LoRA adapters to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Setup optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model_name,
        subfolder="scheduler"
    )
    
    # Load dataset
    train_dataset = MadhubaniDataset(
        data_dir=config.dataset_path,
        tokenizer=tokenizer,
        size=config.resolution,
        center_crop=config.center_crop,
        random_flip=config.random_flip,
        subset_size=args.subset_size,
        seed=config.seed,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move vae and text_encoder to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Training loop
    total_batch_size = (
        config.train_batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Training epochs
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, config.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Gather losses
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {
                    "loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch
                }
                progress_bar.set_postfix(**logs)
                
                if args.use_wandb and accelerator.is_main_process:
                    wandb.log(logs, step=global_step)
                
                train_loss = 0.0
                
                # Save checkpoint
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            if global_step >= config.max_train_steps:
                break
        
        # Epoch end logging
        logger.info(f"Epoch {epoch} completed")
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(config.output_dir, "lora_weights"))
        
        # Save full pipeline for easy inference
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model_name,
            unet=accelerator.unwrap_model(unet),
            revision=config.revision,
        )
        pipeline.save_pretrained(config.output_dir)
        
        logger.info(f"Model saved to {config.output_dir}")
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
