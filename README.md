# Madhubani Art Fine-Tuning with LoRA/DoRA

Hey everyone! So for my deep learning class project, I tackled something really close to my heart, using AI to preserve a 2,500-year-old Indian folk art tradition called Madhubani painting. These are not just pretty pictures, they are intricate geometric masterpieces with distinctive double-line borders and symbolic patterns that general AI models completely mess up.

This project implements parameter-efficient fine-tuning techniques (LoRA, DoRA, HyperLoRA) for generating authentic Madhubani folk art using Stable Diffusion.

In order to create authentic Madhubani paintings using Stable Diffusion models, this project explores parameter-efficient fine-tuning techniques. Three adaptation methods—Low-Rank Adaptation, Weight-Decomposed Low-Rank Adaptation, and HyperLoRA were carefully compared with different rank values and data set sizes.

## Project Overview

Based on the research paper "Fine-Tuning Generative Model for Madhubani Art Synthesis using LoRA", this implementation provides:

- **Multiple Fine-Tuning Methods**: LoRA, DoRA (Weight-Decomposed LoRA), and HyperLoRA
- **Dataset Size Experiments**: Test with 10, 20, or 50 training images
- **Comprehensive Evaluation**: Training curves, loss metrics, and visual quality analysis
- **Interactive Demo**: Gradio web interface for comparing models
- **Production-Ready**: Efficient inference with various schedulers

### Key Results
- **Best Overall Quality**: DoRA r=64 (training loss: 0.1246)
- **Fastest Training**: LoRA r=32 (2.2 minutes on RTX 6000)
- **Minimum Dataset**: 20 images for acceptable results, 50 for professional quality
- **Recommended LoRA Scale**: 1.3-1.5 for inference

## Quick Start

### 1. Installation

```bash
# Clone the repository (or extract files)
cd madhubani-art-finetuning

# Make setup script executable
chmod +x setup.sh

# Run setup (creates conda env and installs dependencies)
./setup.sh
```

Or install manually:
```bash
# Create environment
conda create -n madhubani python=3.10
conda activate madhubani

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset as follows:
```
madhubani_dataset/
├── image_001.jpg
├── image_001.txt
├── image_002.jpg
├── image_002.txt
└── ...
```

Each `.txt` file should contain a caption in this format:
```
madhubani art painting, [subject description], intricate geometric patterns, double-line borders, vibrant red blue yellow green colors, traditional Indian folk art
```

**Example captions:**
```
# For elephant image
madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs

# For peacock image
madhubani art painting, majestic peacock with ornate tail feathers, intricate geometric patterns, bold black outlines, vibrant red yellow blue green colors, traditional indian folk art
```

### 3. Train a Model

**Train LoRA r=32 (fastest):**
```bash
python train_lora.py \
    --config_name lora_r32 \
    --dataset_path ./madhubani_dataset \
    --output_dir ./output_lora_r32
```

**Train DoRA r=64 (best quality):**
```bash
python train_lora.py \
    --config_name dora_r64 \
    --dataset_path ./madhubani_dataset \
    --output_dir ./output_dora_r64
```

**Train with subset (e.g., 20 images):**
```bash
python train_lora.py \
    --config_name lora_r32 \
    --dataset_path ./madhubani_dataset \
    --subset_size 20 \
    --output_dir ./output_lora_r32_subset20
```

**With W&B logging:**
```bash
python train_lora.py \
    --config_name dora_r64 \
    --dataset_path ./madhubani_dataset \
    --use_wandb \
    --wandb_project madhubani-experiments
```

### 4. Generate Images

```bash
python inference.py \
    --model_path ./output_dora_r64 \
    --prompt "madhubani art painting, sacred elephant adorned with traditional jewelry" \
    --num_images 4 \
    --lora_scale 1.3
```

**Use different schedulers:**
```bash
# DPM++ (recommended)
python inference.py --model_path ./output_dora_r64 --scheduler dpm++

# Euler A (faster)
python inference.py --model_path ./output_dora_r64 --scheduler euler_a
```

### 5. Launch Interactive Demo

```bash
python gradio_demo.py \
    --models ./output_lora_r32 ./output_dora_r64 \
    --names "LoRA r=32" "DoRA r=64" \
    --share
```

## Available Configurations

### LoRA Variants
- `lora_r32`: Rank 32 (recommended for quick iteration)
- `lora_r64`: Rank 64 (better quality)
- `lora_r128`: Rank 128 (diminishing returns)

### DoRA Variants
- `dora_r16`: Rank 16 (most efficient)
- `dora_r32`: Rank 32 (balanced)
- `dora_r64`: Rank 64 (best quality)

### HyperLoRA
- `hyperlora_r8`: Rank 8 (experimental, requires more training time)

### Stable Diffusion v3
- `sdv3_base`: For SD v3 experiments

## Advanced Usage

### Custom Training Configuration

Create a custom config in `config.py`:

```python
@dataclass
class CustomConfig(LoRAConfig):
    rank: int = 48
    alpha: int = 48
    learning_rate: float = 3e-5
    num_train_epochs: int = 40
    output_dir: str = "./output_custom"
```

### Dataset Preparation Helper

```python
from dataset import prepare_dataset_structure, create_caption_file

# Organize raw images
prepare_dataset_structure(
    source_dir="./raw_images",
    output_dir="./madhubani_dataset",
    generate_captions=True
)

# Create custom caption
create_caption_file(
    image_path="./madhubani_dataset/my_image.jpg",
    subject="peacock with elaborate tail",
    details="surrounded by flowers and geometric borders",
    colors="vibrant blue, green, yellow, and red"
)
```

### Evaluation and Analysis

```bash
# Analyze single training run
python evaluation.py --log_dir ./output_lora_r32/logs --output training_curve.png

# Compare multiple models
python evaluation.py \
    --compare ./output_lora_r32/logs ./output_dora_r64/logs \
    --compare_names "LoRA r=32" "DoRA r=64" \
    --output comparison.png
```

### Dataset Size Impact Analysis

```python
from evaluation import analyze_dataset_size_impact

analyze_dataset_size_impact(
    results_dir="./outputs",
    method="dora",
    rank=32,
    save_dir="./analysis"
)
```

## Expected Results

### Training Times (on RTX 6000 Ada, 30 epochs, 50 images)
| Method | Rank | Time | Peak VRAM | Model Size |
|--------|------|------|-----------|------------|
| LoRA | 32 | 2.2 min | 8.2 GB | 25 MB |
| LoRA | 64 | 2.2 min | 8.4 GB | 50 MB |
| DoRA | 64 | 3.1 min | 8.5 GB | 52 MB |
| HyperLoRA | 8 | 4.0 min | 10 GB | 210 MB |

### Training Loss
| Method | Best Loss | Final Loss | Stability |
|--------|-----------|------------|-----------|
| LoRA r=32 | 0.1319 | 0.1537 | Good |
| LoRA r=64 | 0.1081 | 0.1583 | Moderate |
| DoRA r=64 | 0.1246 | 0.1415 | Excellent |
| HyperLoRA r=8 | 0.1184 | 0.1651 | Poor |

### Dataset Size Impact (DoRA r=32)
| Dataset Size | Best Loss | Loss Gap | Verdict |
|--------------|-----------|----------|---------|
| 10 images | 0.0653 | 0.1172 | Overfitting |
| 20 images | 0.0798 | 0.0889 | Acceptable |
| 50 images | 0.0945 | 0.0383 | Optimal |

## Inference Tips

### Prompt Engineering
Always include these elements in your prompts:
- **Trigger phrase**: "madhubani art painting" or "madhubani style"
- **Subject**: Specific deity, animal, or scene
- **Style elements**: "intricate geometric patterns", "double-line borders"
- **Colors**: "vibrant red blue yellow green colors"
- **Context**: "traditional Indian folk art"

### Negative Prompts
Recommended negative prompt:
```
blurry, low quality, distorted, western style, realistic photo, modern art, 
3d render, anime, cartoon, sketch
```

### Generation Parameters
- **Steps**: 25-30 (DPM++ scheduler)
- **Guidance Scale**: 7.0-8.0
- **LoRA Scale**: 1.3-1.5 (higher for stronger Madhubani style)
- **Scheduler**: DPM++ 2M Karras (best quality/speed)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in config
config.train_batch_size = 2
config.gradient_accumulation_steps = 8

# Use gradient checkpointing (enabled by default)
config.gradient_checkpointing = True
```

### Poor Quality Results
1. Check LoRA scale (try 1.3-1.5)
2. Use more training images (aim for 50)
3. Train for more epochs (30-40)
4. Try DoRA instead of LoRA
5. Ensure captions include "madhubani art painting" trigger

### Slow Training
1. Enable xformers: `pip install xformers`
2. Use FP16 mixed precision (default)
3. Reduce gradient accumulation steps
4. Use smaller rank (r=16 or r=32)

### Model Not Loading in Inference
```bash
# Check model structure
ls -R ./output_dir

# Should have either:
# - model_index.json (full pipeline)
# - lora_weights/ directory (LoRA only)
# - pytorch_lora_weights.safetensors

# Load explicitly as LoRA
python inference.py --model_path ./output_dir/lora_weights --base_model runwayml/stable-diffusion-v1-5
```

## Project Structure

```
madhubani-art-finetuning/
├── config.py                 # All training configurations
├── dataset.py                # Dataset loading and preprocessing
├── train_lora.py            # Main training script (LoRA/DoRA)
├── train_hyperlora.py       # HyperLoRA training (experimental)
├── inference.py             # Image generation script
├── evaluation.py            # Analysis and evaluation tools
├── gradio_demo.py          # Interactive web interface
├── requirements.txt         # Python dependencies
├── setup.sh                # Installation script
├── README.md               # This file
├── madhubani_dataset/      # Your training images
├── outputs/                # Trained models
├── generated_images/       # Generated outputs
└── logs/                   # Training logs
```

## Recommended Workflows

### Workflow 1: Quick Prototype
```bash
# Train with LoRA r=32 on 20 images
python train_lora.py --config_name lora_r32 --dataset_path ./madhubani_dataset --subset_size 20

# Generate test images
python inference.py --model_path ./output_lora_r32 --num_images 8

# If results are good, train on full dataset
python train_lora.py --config_name lora_r32 --dataset_path ./madhubani_dataset
```

### Workflow 2: Production Quality
```bash
# Train DoRA r=64 on full dataset
python train_lora.py --config_name dora_r64 --dataset_path ./madhubani_dataset --use_wandb

# Evaluate training
python evaluation.py --log_dir ./output_dora_r64/logs

# Generate with optimal settings
python inference.py \
    --model_path ./output_dora_r64 \
    --lora_scale 1.4 \
    --scheduler dpm++ \
    --num_images 16
```

### Workflow 3: Research/Comparison
```bash
# Train multiple configurations
for config in lora_r32 lora_r64 dora_r32 dora_r64; do
    python train_lora.py --config_name $config --dataset_path ./madhubani_dataset --use_wandb
done

# Compare results
python evaluation.py \
    --compare ./output_lora_r32/logs ./output_lora_r64/logs ./output_dora_r32/logs ./output_dora_r64/logs \
    --compare_names "LoRA r=32" "LoRA r=64" "DoRA r=32" "DoRA r=64"

# Launch comparison demo
python gradio_demo.py \
    --models ./output_lora_r32 ./output_lora_r64 ./output_dora_r32 ./output_dora_r64 \
    --names "LoRA r=32" "LoRA r=64" "DoRA r=32" "DoRA r=64"
```

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752)
- [PEFT Library](https://github.com/huggingface/peft)

## Contributing

This implementation is based on the research paper and can be extended with:
- Multi-style learning (different Madhubani sub-styles)
- Human-in-the-loop refinement
- Additional PEFT methods
- Cross-cultural art style transfer

## License

This project is for educational and research purposes. The Madhubani art tradition belongs to the Mithila region and its artists.

## Acknowledgments

- Traditional Mithila artists who created the original art form
- Anthropic for Claude and AI development tools
- HuggingFace for the Diffusers and PEFT libraries
- The open-source AI community
