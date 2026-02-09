"""
Example usage script demonstrating the full workflow
"""

import os
from pathlib import Path

def example_1_basic_training():
    """Example 1: Basic LoRA training"""
    print("="*50)
    print("Example 1: Basic LoRA Training")
    print("="*50)
    
    os.system("""
python train_lora.py \\
    --config_name lora_r32 \\
    --dataset_path ./madhubani_dataset \\
    --output_dir ./output_lora_r32
""")


def example_2_dora_training():
    """Example 2: DoRA training with W&B logging"""
    print("="*50)
    print("Example 2: DoRA Training with W&B")
    print("="*50)
    
    os.system("""
python train_lora.py \\
    --config_name dora_r64 \\
    --dataset_path ./madhubani_dataset \\
    --output_dir ./output_dora_r64 \\
    --use_wandb \\
    --wandb_project madhubani-experiments
""")


def example_3_subset_training():
    """Example 3: Training with dataset subset"""
    print("="*50)
    print("Example 3: Training with 20 Image Subset")
    print("="*50)
    
    os.system("""
python train_lora.py \\
    --config_name lora_r32 \\
    --dataset_path ./madhubani_dataset \\
    --subset_size 20 \\
    --output_dir ./output_lora_r32_subset20
""")


def example_4_inference():
    """Example 4: Generate images"""
    print("="*50)
    print("Example 4: Generate Images")
    print("="*50)
    
    os.system("""
python inference.py \\
    --model_path ./output_dora_r64 \\
    --prompt "madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs" \\
    --num_images 4 \\
    --lora_scale 1.3 \\
    --output_dir ./generated_images/elephant
""")


def example_5_batch_generation():
    """Example 5: Batch generation with different prompts"""
    print("="*50)
    print("Example 5: Batch Generation")
    print("="*50)
    
    prompts = {
        "elephant": "madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs",
        "peacock": "madhubani art painting, majestic peacock with ornate tail feathers, intricate geometric patterns, bold black outlines, vibrant red yellow blue green colors, traditional indian folk art",
        "ganesh": "madhubani art painting, sacred Ganesh with ornate decorations, intricate geometric patterns, double-line borders, vibrant red yellow blue green colors, traditional folk art motifs",
        "tree": "madhubani art painting, Tree of Life with intricate branches and leaves, geometric patterns, bold black borders, vibrant colors, traditional Mithila style",
    }
    
    for name, prompt in prompts.items():
        print(f"\nGenerating {name}...")
        os.system(f"""
python inference.py \\
    --model_path ./output_dora_r64 \\
    --prompt "{prompt}" \\
    --num_images 4 \\
    --lora_scale 1.3 \\
    --output_dir ./generated_images/{name}
""")


def example_6_comparison():
    """Example 6: Compare multiple models"""
    print("="*50)
    print("Example 6: Model Comparison")
    print("="*50)
    
    os.system("""
python evaluation.py \\
    --compare ./output_lora_r32/logs ./output_dora_r64/logs \\
    --compare_names "LoRA r=32" "DoRA r=64" \\
    --output ./comparison_plot.png
""")


def example_7_launch_demo():
    """Example 7: Launch Gradio demo"""
    print("="*50)
    print("Example 7: Launch Interactive Demo")
    print("="*50)
    
    os.system("""
python gradio_demo.py \\
    --models ./output_lora_r32 ./output_dora_r64 \\
    --names "LoRA r=32" "DoRA r=64" \\
    --share
""")


def example_8_custom_config():
    """Example 8: Use custom configuration"""
    print("="*50)
    print("Example 8: Custom Configuration")
    print("="*50)
    
    from config import LoRAConfig
    from dataclasses import dataclass
    
    @dataclass
    class MyCustomConfig(LoRAConfig):
        rank: int = 48
        alpha: int = 48
        learning_rate: float = 3e-5
        num_train_epochs: int = 40
        train_batch_size: int = 2
        gradient_accumulation_steps: int = 8
        output_dir: str = "./output_custom"
    
    # Save config and use it
    config = MyCustomConfig()
    print(f"Custom config: rank={config.rank}, lr={config.learning_rate}")


def example_9_dataset_preparation():
    """Example 9: Prepare dataset from raw images"""
    print("="*50)
    print("Example 9: Dataset Preparation")
    print("="*50)
    
    from dataset import prepare_dataset_structure, create_caption_file
    
    # Organize raw images
    prepare_dataset_structure(
        source_dir="./raw_images",
        output_dir="./madhubani_dataset",
        generate_captions=True
    )
    
    # Create custom captions for specific images
    create_caption_file(
        image_path="./madhubani_dataset/peacock_001.jpg",
        subject="majestic peacock with elaborate tail",
        details="surrounded by flowers and geometric borders, intricate patterns",
        colors="vibrant blue, green, yellow, and red"
    )


def example_10_evaluation():
    """Example 10: Detailed evaluation"""
    print("="*50)
    print("Example 10: Detailed Evaluation")
    print("="*50)
    
    from evaluation import TrainingAnalyzer, analyze_dataset_size_impact
    
    # Analyze single run
    analyzer = TrainingAnalyzer("./output_dora_r64/logs")
    analyzer.print_summary()
    analyzer.plot_training_curve(save_path="training_curve.png")
    
    # Analyze dataset size impact
    analyze_dataset_size_impact(
        results_dir="./experiments",
        method="dora",
        rank=32,
        save_dir="./analysis"
    )


def main():
    """Main menu"""
    examples = {
        "1": ("Basic LoRA Training", example_1_basic_training),
        "2": ("DoRA Training with W&B", example_2_dora_training),
        "3": ("Training with Subset", example_3_subset_training),
        "4": ("Generate Images", example_4_inference),
        "5": ("Batch Generation", example_5_batch_generation),
        "6": ("Compare Models", example_6_comparison),
        "7": ("Launch Demo", example_7_launch_demo),
        "8": ("Custom Configuration", example_8_custom_config),
        "9": ("Dataset Preparation", example_9_dataset_preparation),
        "10": ("Evaluation", example_10_evaluation),
    }
    
    print("\n" + "="*50)
    print("Madhubani Art Fine-Tuning - Examples")
    print("="*50)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Run all examples")
    print("  q. Quit")
    
    choice = input("\nSelect example (1-10, 0, or q): ").strip()
    
    if choice == "q":
        return
    elif choice == "0":
        for _, (_, func) in sorted(examples.items()):
            func()
            input("\nPress Enter to continue...")
    elif choice in examples:
        _, func = examples[choice]
        func()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
