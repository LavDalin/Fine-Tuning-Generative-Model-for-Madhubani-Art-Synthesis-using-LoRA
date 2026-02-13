"""
Gradio web interface for comparing Madhubani art generation across models
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import time
from pathlib import Path
from typing import List, Dict, Optional
import os


class MadhubaniGenerator:
    """Wrapper class for managing multiple fine-tuned models"""
    
    def __init__(self):
        self.pipelines = {}
        self.base_model = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    def load_model(self, model_name: str, model_path: str):
        """Load a fine-tuned model"""
        if model_name in self.pipelines:
            return
        
        print(f"Loading {model_name} from {model_path}...")
        
        try:
            # Load base pipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=self.dtype,
                safety_checker=None,
            )
            
            # Load LoRA weights
            lora_path = Path(model_path)
            if (lora_path / "pytorch_lora_weights.safetensors").exists():
                pipe.load_lora_weights(lora_path)
            elif (lora_path / "lora_weights").exists():
                pipe.load_lora_weights(lora_path / "lora_weights")
            
            # Setup scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            
            # Move to device
            pipe = pipe.to(self.device)
            
            # Enable memory optimizations
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            self.pipelines[model_name] = pipe
            print(f"Loaded {model_name} successfully")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    def generate(
        self,
        model_name: str,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        lora_scale: float,
        seed: int,
    ) -> tuple:
        """Generate an image with specified parameters"""
        
        if model_name not in self.pipelines:
            return None, "Model not loaded"
        
        pipe = self.pipelines[model_name]
        
        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        start_time = time.time()
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs={"scale": lora_scale},
            ).images[0]
            
            generation_time = time.time() - start_time
            
            info = f"Generated in {generation_time:.2f}s"
            return image, info
            
        except Exception as e:
            return None, f"Error: {str(e)}"


# Global generator instance
generator = MadhubaniGenerator()


# Predefined prompts
EXAMPLE_PROMPTS = {
    "Elephant": "madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs",
    "Peacock": "madhubani art painting, majestic peacock with ornate tail feathers, intricate geometric patterns, bold black outlines, vibrant red yellow blue green colors, traditional indian folk art",
    "Ganesh": "madhubani art painting, sacred Ganesh with ornate decorations, intricate geometric patterns, double-line borders, vibrant red yellow blue green colors, traditional folk art motifs",
    "Tree of Life": "madhubani art painting, Tree of Life with intricate branches and leaves, geometric patterns, bold black borders, vibrant colors, traditional Mithila style",
    "Krishna": "madhubani art painting, Lord Krishna playing flute, surrounded by peacocks and flowers, intricate patterns, double-line borders, vibrant traditional colors",
    "Fish": "madhubani art painting, pair of fish swimming in circular pattern, intricate scales, geometric patterns, bold outlines, vibrant red blue yellow green, traditional folk motifs",
}


def create_demo(model_configs: Dict[str, str]):
    """
    Create Gradio interface
    
    Args:
        model_configs: Dictionary mapping model names to paths
    """
    
    # Load models
    for name, path in model_configs.items():
        generator.load_model(name, path)
    
    model_names = list(model_configs.keys())
    
    def generate_image(
        model_name,
        prompt,
        negative_prompt,
        steps,
        guidance,
        lora_scale,
        seed,
    ):
        return generator.generate(
            model_name,
            prompt,
            negative_prompt,
            steps,
            guidance,
            lora_scale,
            seed,
        )
    
    def generate_comparison(
        prompt,
        negative_prompt,
        steps,
        guidance,
        lora_scale,
        seed,
    ):
        """Generate images from all loaded models"""
        results = []
        
        for model_name in model_names:
            image, info = generator.generate(
                model_name,
                prompt,
                negative_prompt,
                steps,
                guidance,
                lora_scale,
                seed,
            )
            results.append((image, f"{model_name}\n{info}"))
        
        return results
    
    def load_example_prompt(example_name):
        return EXAMPLE_PROMPTS.get(example_name, "")
    
    # Create interface
    with gr.Blocks(title="Madhubani Art Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Madhubani Art Generator
            ### Fine-tuned Stable Diffusion models for authentic Madhubani folk art
            
            Compare different fine-tuning methods (LoRA, DoRA, HyperLoRA) for generating traditional Indian Madhubani paintings.
            """
        )
        
        with gr.Tabs():
            # Single Model Generation Tab
            with gr.Tab("Single Model"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_select = gr.Dropdown(
                            choices=model_names,
                            value=model_names[0] if model_names else None,
                            label="Select Model",
                        )
                        
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            value=EXAMPLE_PROMPTS["Elephant"],
                            lines=3,
                        )
                        
                        example_select = gr.Dropdown(
                            choices=list(EXAMPLE_PROMPTS.keys()),
                            label="Load Example Prompt",
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted, western style, realistic photo, modern art",
                            lines=2,
                        )
                        
                        with gr.Row():
                            steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                            guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                        
                        with gr.Row():
                            lora_scale = gr.Slider(0.5, 2.0, value=1.3, step=0.1, label="LoRA Scale")
                            seed = gr.Number(value=42, label="Seed", precision=0)
                        
                        generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Image")
                        output_info = gr.Textbox(label="Info")
                
                # Connect example selector
                example_select.change(
                    fn=load_example_prompt,
                    inputs=[example_select],
                    outputs=[prompt_input],
                )
                
                # Connect generate button
                generate_btn.click(
                    fn=generate_image,
                    inputs=[
                        model_select,
                        prompt_input,
                        negative_prompt,
                        steps,
                        guidance,
                        lora_scale,
                        seed,
                    ],
                    outputs=[output_image, output_info],
                )
            
            # Model Comparison Tab
            with gr.Tab("Compare Models"):
                with gr.Column():
                    gr.Markdown("### Generate and compare outputs from all loaded models")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            comp_prompt = gr.Textbox(
                                label="Prompt",
                                value=EXAMPLE_PROMPTS["Peacock"],
                                lines=3,
                            )
                            
                            comp_example = gr.Dropdown(
                                choices=list(EXAMPLE_PROMPTS.keys()),
                                label="Load Example Prompt",
                            )
                            
                            comp_negative = gr.Textbox(
                                label="Negative Prompt",
                                value="blurry, low quality, distorted, western style, realistic photo, modern art",
                                lines=2,
                            )
                            
                            with gr.Row():
                                comp_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                                comp_guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance")
                            
                            with gr.Row():
                                comp_lora_scale = gr.Slider(0.5, 2.0, value=1.3, step=0.1, label="LoRA Scale")
                                comp_seed = gr.Number(value=42, label="Seed", precision=0)
                            
                            compare_btn = gr.Button("Generate All", variant="primary")
                    
                    comp_gallery = gr.Gallery(
                        label="Comparison Results",
                        columns=min(len(model_names), 4),
                        rows=max(1, len(model_names) // 4 + 1),
                        height="auto",
                    )
                    
                    # Connect example selector
                    comp_example.change(
                        fn=load_example_prompt,
                        inputs=[comp_example],
                        outputs=[comp_prompt],
                    )
                    
                    # Connect compare button
                    compare_btn.click(
                        fn=generate_comparison,
                        inputs=[
                            comp_prompt,
                            comp_negative,
                            comp_steps,
                            comp_guidance,
                            comp_lora_scale,
                            comp_seed,
                        ],
                        outputs=[comp_gallery],
                    )
            
            # About Tab
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About Madhubani Art
                    
                    Madhubani painting, also known as Mithila art, is a traditional Indian folk art 
                    that originated in the Mithila region of Bihar, India, over 2,500 years ago.
                    
                    ### Key Characteristics:
                    - **Double-line borders**: Distinctive thick outer and thin inner lines
                    - **Geometric precision**: Intricate patterns with mathematical symmetry
                    - **Natural pigments**: Colors traditionally derived from plants and minerals
                    - **Symbolic motifs**: Depictions of Hindu deities, nature, and everyday life
                    
                    ### Fine-Tuning Methods:
                    
                    **LoRA (Low-Rank Adaptation)**:
                    - Fast training and inference
                    - Parameter efficient (< 1% trainable parameters)
                    - Good balance of quality and speed
                    
                    **DoRA (Weight-Decomposed LoRA)**:
                    - Separates magnitude and direction learning
                    - Better pattern recognition
                    - More stable training
                    - Best overall quality for Madhubani art
                    
                    **HyperLoRA**:
                    - Dynamic weight generation
                    - Context-aware adaptation
                    - Experimental (longer training time)
                    
                    ### Recommendations:
                    - **For production**: DoRA r=64 (best quality)
                    - **For quick iteration**: LoRA r=32 (fastest)
                    - **LoRA Scale**: 1.3-1.5 for authentic style
                    - **Dataset**: Minimum 20 images, optimal at 50 images
                    """
                )
    
    return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio demo for Madhubani art generation")
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        help="Paths to model directories (e.g., ./output_lora_r32 ./output_dora_r64)",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs='+',
        help="Names for the models (optional, will use paths if not provided)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    
    args = parser.parse_args()
    
    # Create model configs
    if args.names and len(args.names) == len(args.models):
        model_configs = dict(zip(args.names, args.models))
    else:
        model_configs = {Path(p).name: p for p in args.models}
    
    print("="*50)
    print("Madhubani Art Generator - Gradio Demo")
    print("="*50)
    print(f"Loading {len(model_configs)} models:")
    for name, path in model_configs.items():
        print(f"  - {name}: {path}")
    print("="*50)
    
    # Create and launch demo
    demo = create_demo(model_configs)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
