import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MadhubaniDataset(Dataset):
    """Dataset class for Madhubani art images with captions"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        size: int = 512,
        center_crop: bool = True,
        random_flip: bool = False,
        subset_size: Optional[int] = None,
        stratified: bool = True,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing images and caption files
            tokenizer: CLIP tokenizer for text encoding
            size: Target image size
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
            subset_size: If provided, use only this many images
            stratified: If True, use stratified sampling for subsets
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        
        # Load image and caption pairs
        self.image_paths, self.captions = self._load_dataset()
        
        # Create subset if requested
        if subset_size and subset_size < len(self.image_paths):
            self._create_subset(subset_size, stratified, seed)
        
        # Setup transforms
        self.transforms = self._setup_transforms(center_crop, random_flip)
        
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
    
    def _load_dataset(self) -> Tuple[List[Path], List[str]]:
        """Load all image-caption pairs from the dataset directory"""
        image_paths = []
        captions = []
        
        # Support both .txt files and metadata.json
        for img_path in sorted(self.data_dir.glob("*.jpg")) + sorted(self.data_dir.glob("*.png")):
            # Try to load caption from .txt file
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                # Default caption if no caption file exists
                caption = "madhubani art painting, traditional Indian folk art"
            
            image_paths.append(img_path)
            captions.append(caption)
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        return image_paths, captions
    
    def _create_subset(self, subset_size: int, stratified: bool, seed: int):
        """Create a subset of the dataset"""
        random.seed(seed)
        
        if stratified:
            # Try to maintain distribution of different subjects
            # This is a simple version - you can enhance this based on your metadata
            indices = list(range(len(self.image_paths)))
            random.shuffle(indices)
            selected_indices = indices[:subset_size]
        else:
            # Random sampling
            selected_indices = random.sample(range(len(self.image_paths)), subset_size)
        
        self.image_paths = [self.image_paths[i] for i in selected_indices]
        self.captions = [self.captions[i] for i in selected_indices]
        
        print(f"Created subset of {subset_size} images")
    
    def _setup_transforms(self, center_crop: bool, random_flip: bool):
        """Setup image preprocessing transforms"""
        transform_list = []
        
        if center_crop:
            transform_list.append(transforms.CenterCrop(self.size))
        else:
            transform_list.append(transforms.Resize(self.size))
        
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Resize to target resolution
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Apply transforms
        image = self.transforms(image)
        
        # Tokenize caption
        caption = self.captions[idx]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids[0],
            "caption": caption
        }


def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }


def create_caption_file(image_path: str, subject: str, details: str, colors: str) -> str:
    """
    Helper function to generate standardized Madhubani captions
    
    Args:
        image_path: Path to the image file
        subject: Main subject (e.g., "sacred Ganesh", "peacock", "Tree of Life")
        details: Specific details about the subject
        colors: Color description
    
    Returns:
        Formatted caption string
    """
    caption = (
        f"madhubani art painting, {subject}, {details}, "
        f"intricate geometric patterns, double-line borders, "
        f"{colors}, traditional Indian folk art, "
        f"bold black outlines, vibrant colors"
    )
    
    # Save to .txt file
    txt_path = Path(image_path).with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(caption)
    
    return caption


def prepare_dataset_structure(
    source_dir: str,
    output_dir: str,
    generate_captions: bool = False
):
    """
    Prepare dataset in the correct structure
    
    Args:
        source_dir: Directory containing source images
        output_dir: Output directory for organized dataset
        generate_captions: If True, generate default captions for images without captions
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy images and create captions if needed
    for img_path in source_path.glob("*.jpg") + source_path.glob("*.png"):
        # Copy image
        output_img = output_path / img_path.name
        if not output_img.exists():
            import shutil
            shutil.copy(img_path, output_img)
        
        # Check for caption
        txt_path = img_path.with_suffix('.txt')
        output_txt = output_path / txt_path.name
        
        if txt_path.exists() and not output_txt.exists():
            import shutil
            shutil.copy(txt_path, output_txt)
        elif generate_captions and not output_txt.exists():
            # Generate default caption
            caption = (
                "madhubani art painting, traditional Indian folk art, "
                "intricate geometric patterns, double-line borders, "
                "vibrant colors, bold black outlines"
            )
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(caption)
            print(f"Generated caption for {img_path.name}")
    
    print(f"Dataset prepared in {output_dir}")


if __name__ == "__main__":
    # Example usage
    from transformers import CLIPTokenizer
    
    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )
    
    # Create dataset
    dataset = MadhubaniDataset(
        data_dir="./madhubani_dataset",
        tokenizer=tokenizer,
        size=512,
        subset_size=20  # Use subset of 20 images
    )
    
    # Test loading
    sample = dataset[0]
    print(f"Sample shape: {sample['pixel_values'].shape}")
    print(f"Caption: {sample['caption']}")
