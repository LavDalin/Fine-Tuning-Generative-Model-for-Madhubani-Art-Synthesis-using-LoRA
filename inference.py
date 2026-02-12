"""
Inference script for generating Madhubani art with fine-tuned models
"""

import argparse
import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Madhubani art with fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model or LoRA weights",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, distorted, western style, realistic photo, modern art",
        help="Negative prompt",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.3,
        help="LoRA weight scale (recommended 1.3-1.5 for Madhubani)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm++",
        choices=["dpm++", "ddim", "pndm", "euler", "euler_a"],
        help="Scheduler to use for inference",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 for faster inference",
    )
    
    return parser.parse_args()


def load_pipeline(model_path: str, base_model: str, scheduler: str, use_fp16: bool = False):
    """Load the inference pipeline"""
    
    # Determine dtype
    dtype = torch.float16 if use_fp16 else torch.float32
    
    # Check if model_path contains full pipeline or just LoRA weights
    model_path = Path(model_path)
    
    if (model_path / "model_index.json").exists():
        # Full pipeline saved
        print(f"Loading full pipeline from {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
        )
    else:
        # Load base model and LoRA weights
        print(f"Loading base model {base_model} with LoRA weights from {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None,
        )
        
        # Load LoRA weights
        if (model_path / "pytorch_lora_weights.safetensors").exists():
            pipe.load_lora_weights(model_path)
        elif (model_path / "lora_weights").exists():
            pipe.load_lora_weights(model_path / "lora_weights")
        else:
            print(f"Warning: Could not find LoRA weights in {model_path}")
    
    # Setup scheduler
    if scheduler == "dpm++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    elif scheduler == "euler":
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "euler_a":
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using GPU for inference")
    else:
        print("Using CPU for inference (this will be slow)")
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except:
        print("xformers not available, using default attention")
    
    return pipe


def generate_images(
    pipe,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    guidance_scale: float,
    num_inference_steps: int,
    lora_scale: float,
    seed: int,
    output_dir: str,
):
    """Generate images with the pipeline"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    
    images = []
    generation_times = []
    
    for i in range(num_images):
        print(f"\nGenerating image {i+1}/{num_images}...")
        
        # Set seed (different for each image)
        current_seed = seed + i
        generator.manual_seed(current_seed)
        
        # Generate
        start_time = time.time()
        
        if hasattr(pipe, 'unet') and hasattr(pipe.unet, 'set_adapters'):
            # For LoRA models, set the scale
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs={"scale": lora_scale},
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        
        # Save image
        output_path = os.path.join(output_dir, f"image_{i+1}_seed{current_seed}.png")
        image.save(output_path)
        images.append(image)
        
        print(f"Saved to {output_path} (generated in {generation_time:.2f}s)")
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total images: {num_images}")
    print(f"Average generation time: {sum(generation_times)/len(generation_times):.2f}s")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")
    
    return images


def main():
    args = parse_args()
    
    print("="*50)
    print("Madhubani Art Generator")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Number of images: {args.num_images}")
    print(f"LoRA scale: {args.lora_scale}")
    print("="*50)
    
    # Load pipeline
    pipe = load_pipeline(
        args.model_path,
        args.base_model,
        args.scheduler,
        args.use_fp16
    )
    
    # Generate images
    generate_images(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.num_images,
        args.guidance_scale,
        args.num_inference_steps,
        args.lora_scale,
        args.seed,
        args.output_dir,
    )


# Predefined prompts for testing
MADHUBANI_PROMPTS = {
    "elephant": "madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs",
    "peacock": "madhubani art painting, majestic peacock with ornate tail feathers, intricate geometric patterns, bold black outlines, vibrant red yellow blue green colors, traditional indian folk art",
    "ganesh": "madhubani art painting, sacred Ganesh with ornate decorations, intricate geometric patterns, double-line borders, vibrant red yellow blue green colors, traditional folk art motifs",
    "tree_of_life": "madhubani art painting, Tree of Life with intricate branches and leaves, geometric patterns, bold black borders, vibrant colors, traditional Mithila style",
    "krishna": "madhubani art painting, Lord Krishna playing flute, surrounded by peacocks and flowers, intricate patterns, double-line borders, vibrant traditional colors",
    "chhath_puja": "madhubani style, Chhath Puja ritual scene, women worshiping the Sun God standing in river with offerings, surrounded by sugarcane and banana plants, bright colors, geometric and patterned borders",
    "fish": "madhubani art painting, pair of fish swimming in circular pattern, intricate scales, geometric patterns, bold outlines, vibrant red blue yellow green, traditional folk motifs",
}


if __name__ == "__main__":
    main()
