"""
Configuration file for Madhubani Art Fine-Tuning
Contains all hyperparameters and settings for LoRA, DoRA, and HyperLoRA training
"""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    variant: Optional[str] = None
    
    # LoRA specific parameters
    rank: int = 32
    alpha: int = 32  # Usually set equal to rank
    target_modules: List[str] = None
    lora_dropout: float = 0.0
    bias: str = "none"
    
    # Training parameters
    learning_rate: float = 5e-5
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 30
    max_train_steps: Optional[int] = None
    
    # Optimizer settings
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduler
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    
    # Mixed precision
    mixed_precision: str = "fp16"
    
    # Dataset settings
    dataset_path: str = "./madhubani_dataset"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False
    
    # Checkpoint settings
    output_dir: str = "./output_lora"
    checkpointing_steps: int = 500
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"
    
    # Validation
    validation_prompt: str = "madhubani art painting, sacred elephant adorned with traditional jewelry"
    num_validation_images: int = 4
    validation_epochs: int = 5
    
    # Miscellaneous
    seed: int = 42
    gradient_checkpointing: bool = True
    allow_tf32: bool = True
    dataloader_num_workers: int = 0
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        self.effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps


@dataclass
class DoRAConfig(LoRAConfig):
    """Configuration for DoRA fine-tuning (extends LoRA)"""
    # DoRA uses weight decomposition
    use_dora: bool = True
    
    # Different scheduler for DoRA
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    
    output_dir: str = "./output_dora"


@dataclass
class HyperLoRAConfig(LoRAConfig):
    """Configuration for HyperLoRA fine-tuning"""
    # HyperLoRA specific
    rank: int = 8  # Lower rank for HyperLoRA
    
    # Hypernetwork settings
    hypernetwork_hidden_size: int = 256
    hypernetwork_num_layers: int = 2
    
    # Training parameters (HyperLoRA needs more careful training)
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    
    # Scheduler
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 200
    
    # Full precision for stability
    mixed_precision: str = "no"
    
    output_dir: str = "./output_hyperlora"


@dataclass
class SDv3Config(LoRAConfig):
    """Configuration for Stable Diffusion v3"""
    model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resolution: int = 1024
    output_dir: str = "./output_sdv3"


@dataclass
class DatasetConfig:
    """Configuration for dataset experiments"""
    full_dataset_size: int = 50
    subset_sizes: List[int] = None
    stratified_sampling: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        if self.subset_sizes is None:
            self.subset_sizes = [10, 20, 50]


# Preset configurations for different experiments
PRESET_CONFIGS = {
    "lora_r32": LoRAConfig(rank=32, alpha=32, output_dir="./output_lora_r32"),
    "lora_r64": LoRAConfig(rank=64, alpha=64, output_dir="./output_lora_r64"),
    "lora_r128": LoRAConfig(rank=128, alpha=128, output_dir="./output_lora_r128"),
    
    "dora_r16": DoRAConfig(rank=16, alpha=16, output_dir="./output_dora_r16"),
    "dora_r32": DoRAConfig(rank=32, alpha=32, output_dir="./output_dora_r32"),
    "dora_r64": DoRAConfig(rank=64, alpha=64, output_dir="./output_dora_r64"),
    
    "hyperlora_r8": HyperLoRAConfig(rank=8, alpha=8, output_dir="./output_hyperlora_r8"),
    
    "sdv3_base": SDv3Config(output_dir="./output_sdv3"),
}


def get_config(config_name: str):
    """Get a preset configuration by name"""
    if config_name in PRESET_CONFIGS:
        return PRESET_CONFIGS[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(PRESET_CONFIGS.keys())}")
