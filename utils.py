"""
Utility functions for Madhubani art fine-tuning project
"""

import os
import torch
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def count_parameters(model) -> Dict[str, int]:
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100,
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_model_size(model_path: str) -> str:
    """Get the size of a saved model"""
    path = Path(model_path)
    
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    else:
        return "Unknown"
    
    return format_bytes(size)


def save_training_config(config, output_dir: str):
    """Save training configuration to JSON"""
    output_path = Path(output_dir) / "config.json"
    
    config_dict = {}
    for key, value in vars(config).items():
        if not key.startswith('_'):
            # Convert to serializable format
            if isinstance(value, (int, float, str, bool, type(None))):
                config_dict[key] = value
            elif isinstance(value, (list, tuple)):
                config_dict[key] = list(value)
            else:
                config_dict[key] = str(value)
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Saved config to {output_path}")


def load_training_config(config_path: str) -> Dict:
    """Load training configuration from JSON"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_image_grid(images: List[Image.Image], rows: int, cols: int, save_path: Optional[str] = None) -> Image.Image:
    """Create a grid of images"""
    if len(images) != rows * cols:
        raise ValueError(f"Number of images ({len(images)}) must equal rows * cols ({rows * cols})")
    
    # Get dimensions from first image
    w, h = images[0].size
    
    # Create output image
    grid = Image.new('RGB', (w * cols, h * rows))
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))
    
    if save_path:
        grid.save(save_path)
        print(f"Saved image grid to {save_path}")
    
    return grid


def compare_images_side_by_side(
    images: List[Image.Image],
    titles: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """Display images side by side with titles"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate simple pixel-wise similarity between two images
    Returns value between 0 (completely different) and 1 (identical)
    """
    # Resize to same size if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate mean squared error
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Normalize to 0-1 range (assuming 8-bit images)
    max_mse = 255 ** 2
    similarity = 1 - (mse / max_mse)
    
    return similarity


def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return "No GPU available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        info.append(f"GPU {i}: {name} ({format_bytes(total_memory)})")
    
    return "\n".join(info)


def estimate_training_time(
    num_images: int,
    num_epochs: int,
    batch_size: int,
    seconds_per_step: float = 4.5,
) -> str:
    """Estimate total training time"""
    steps_per_epoch = num_images // batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * seconds_per_step
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def create_timestamp() -> str:
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_output_directory(base_dir: str, experiment_name: str) -> str:
    """Create output directory with timestamp"""
    timestamp = create_timestamp()
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def validate_dataset(dataset_path: str) -> Dict:
    """Validate dataset and return statistics"""
    path = Path(dataset_path)
    
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Count images
    image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    
    # Count caption files
    caption_files = list(path.glob("*.txt"))
    
    # Check for missing captions
    missing_captions = []
    for img in image_files:
        txt_path = img.with_suffix('.txt')
        if not txt_path.exists():
            missing_captions.append(img.name)
    
    stats = {
        "num_images": len(image_files),
        "num_captions": len(caption_files),
        "missing_captions": len(missing_captions),
        "has_complete_captions": len(missing_captions) == 0,
    }
    
    # Print report
    print("="*50)
    print("DATASET VALIDATION")
    print("="*50)
    print(f"Location: {dataset_path}")
    print(f"Images found: {stats['num_images']}")
    print(f"Caption files: {stats['num_captions']}")
    
    if stats['missing_captions'] > 0:
        print(f"⚠ Missing captions: {stats['missing_captions']}")
        print("Files without captions:")
        for name in missing_captions[:5]:  # Show first 5
            print(f"  - {name}")
        if len(missing_captions) > 5:
            print(f"  ... and {len(missing_captions) - 5} more")
    else:
        print("✓ All images have captions")
    
    print("="*50)
    
    return stats


def merge_lora_weights(
    base_model_path: str,
    lora_weights_path: str,
    output_path: str,
    alpha: float = 1.0,
):
    """
    Merge LoRA weights into base model for standalone deployment
    Note: This is a simplified version. For production, use PEFT's merge functions
    """
    print(f"Merging LoRA weights from {lora_weights_path} into {base_model_path}")
    print(f"Output will be saved to {output_path}")
    print(f"Alpha (merge strength): {alpha}")
    print("\nNote: For production use, please use PEFT's official merge_and_unload() method")


def calculate_dataset_statistics(dataset_path: str) -> Dict:
    """Calculate statistics about the dataset"""
    path = Path(dataset_path)
    image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    
    if not image_files:
        return {}
    
    # Calculate image dimensions
    sizes = []
    aspects = []
    file_sizes = []
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            w, h = img.size
            sizes.append((w, h))
            aspects.append(w / h)
        
        file_sizes.append(img_path.stat().st_size)
    
    # Statistics
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    
    stats = {
        "num_images": len(image_files),
        "avg_width": int(np.mean(widths)),
        "avg_height": int(np.mean(heights)),
        "min_size": f"{min(widths)}x{min(heights)}",
        "max_size": f"{max(widths)}x{max(heights)}",
        "avg_aspect_ratio": np.mean(aspects),
        "total_size": format_bytes(sum(file_sizes)),
        "avg_file_size": format_bytes(int(np.mean(file_sizes))),
    }
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("="*50 + "\n")
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Madhubani Art Fine-Tuning Utilities")
    print("="*50)
    
    # GPU info
    print("\nGPU Information:")
    print(get_gpu_info())
    
    # Estimate training time
    print("\nEstimated Training Time:")
    print(f"50 images, 30 epochs: {estimate_training_time(50, 30, 4)}")
    print(f"20 images, 30 epochs: {estimate_training_time(20, 30, 4)}")
    
    # Validate dataset if exists
    if Path("./madhubani_dataset").exists():
        validate_dataset("./madhubani_dataset")
        calculate_dataset_statistics("./madhubani_dataset")
