#!/bin/bash

# Script to replicate all experiments from the paper
# "Fine-Tuning Generative Model for Madhubani Art Synthesis using LoRA"

echo "=================================================="
echo "Madhubani Art Fine-Tuning - Full Experiment Suite"
echo "=================================================="

# Configuration
DATASET_PATH="./madhubani_dataset"
BASE_OUTPUT="./experiments"
USE_WANDB="--use_wandb --wandb_project madhubani-experiments"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory '$DATASET_PATH' not found"
    echo "Please place your Madhubani images in this directory"
    exit 1
fi

# Count images
num_images=$(find "$DATASET_PATH" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "Found $num_images images in dataset"

if [ "$num_images" -lt 10 ]; then
    echo "Warning: Less than 10 images found. You need at least 10 images."
    exit 1
fi

# Create output directory
mkdir -p "$BASE_OUTPUT"

# Function to train a model
train_model() {
    local config=$1
    local subset=$2
    local output_dir="$BASE_OUTPUT/${config}"
    
    if [ ! -z "$subset" ]; then
        output_dir="${output_dir}_subset${subset}"
    fi
    
    echo ""
    echo "=================================================="
    echo "Training: $config $([ ! -z "$subset" ] && echo "with $subset images")"
    echo "Output: $output_dir"
    echo "=================================================="
    
    if [ ! -z "$subset" ]; then
        python train_lora.py \
            --config_name "$config" \
            --dataset_path "$DATASET_PATH" \
            --output_dir "$output_dir" \
            --subset_size "$subset" \
            $USE_WANDB
    else
        python train_lora.py \
            --config_name "$config" \
            --dataset_path "$DATASET_PATH" \
            --output_dir "$output_dir" \
            $USE_WANDB
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully"
    else
        echo "✗ Training failed"
        return 1
    fi
}

# ============================================
# Experiment 1: Method Comparison (Full Dataset)
# ============================================
echo ""
echo "###############################################"
echo "# EXPERIMENT 1: Method Comparison"
echo "###############################################"

# LoRA variants
train_model "lora_r32"
train_model "lora_r64"
train_model "lora_r128"

# DoRA variants
train_model "dora_r16"
train_model "dora_r32"
train_model "dora_r64"

# HyperLoRA (if you want to run it - takes longer)
read -p "Train HyperLoRA? (takes ~4 min, experimental) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    train_model "hyperlora_r8"
fi

# ============================================
# Experiment 2: Dataset Size Impact
# ============================================
echo ""
echo "###############################################"
echo "# EXPERIMENT 2: Dataset Size Impact"
echo "###############################################"

# Only run if we have enough images
if [ "$num_images" -ge 50 ]; then
    # LoRA r=32 with different dataset sizes
    train_model "lora_r32" 10
    train_model "lora_r32" 20
    train_model "lora_r32" 50
    
    # DoRA r=32 with different dataset sizes
    train_model "dora_r32" 10
    train_model "dora_r32" 20
    train_model "dora_r32" 50
else
    echo "Skipping dataset size experiments (need 50+ images, have $num_images)"
fi

# ============================================
# Experiment 3: Generate Comparison Images
# ============================================
echo ""
echo "###############################################"
echo "# EXPERIMENT 3: Generate Test Images"
echo "###############################################"

# Standard prompts from the paper
ELEPHANT_PROMPT="madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs"
PEACOCK_PROMPT="madhubani art painting, majestic peacock with ornate tail feathers, intricate geometric patterns, bold black outlines, vibrant red yellow blue green colors, traditional indian folk art"
GANESH_PROMPT="madhubani art painting, sacred Ganesh with ornate decorations, intricate geometric patterns, double-line borders, vibrant red yellow blue green colors, traditional folk art motifs"

# Generate with best models
for model in "$BASE_OUTPUT/lora_r32" "$BASE_OUTPUT/lora_r64" "$BASE_OUTPUT/dora_r64"; do
    if [ -d "$model" ]; then
        model_name=$(basename "$model")
        output_dir="$BASE_OUTPUT/generated_${model_name}"
        
        echo "Generating images with $model_name..."
        
        python inference.py \
            --model_path "$model" \
            --prompt "$ELEPHANT_PROMPT" \
            --num_images 4 \
            --output_dir "$output_dir/elephant" \
            --lora_scale 1.3
        
        python inference.py \
            --model_path "$model" \
            --prompt "$PEACOCK_PROMPT" \
            --num_images 4 \
            --output_dir "$output_dir/peacock" \
            --lora_scale 1.3
        
        python inference.py \
            --model_path "$model" \
            --prompt "$GANESH_PROMPT" \
            --num_images 4 \
            --output_dir "$output_dir/ganesh" \
            --lora_scale 1.3
    fi
done

# ============================================
# Experiment 4: Generate Analysis Plots
# ============================================
echo ""
echo "###############################################"
echo "# EXPERIMENT 4: Analysis and Visualization"
echo "###############################################"

# Compare methods
python evaluation.py \
    --compare \
        "$BASE_OUTPUT/lora_r32/logs" \
        "$BASE_OUTPUT/lora_r64/logs" \
        "$BASE_OUTPUT/dora_r64/logs" \
    --compare_names "LoRA r=32" "LoRA r=64" "DoRA r=64" \
    --output "$BASE_OUTPUT/method_comparison.png"

# Analyze dataset size impact (if experiments were run)
if [ "$num_images" -ge 50 ]; then
    python -c "
from evaluation import analyze_dataset_size_impact
analyze_dataset_size_impact(
    results_dir='$BASE_OUTPUT',
    method='lora',
    rank=32,
    save_dir='$BASE_OUTPUT/analysis'
)
analyze_dataset_size_impact(
    results_dir='$BASE_OUTPUT',
    method='dora',
    rank=32,
    save_dir='$BASE_OUTPUT/analysis'
)
"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=================================================="
echo ""
echo "Results saved to: $BASE_OUTPUT/"
echo ""
echo "Key findings:"
echo "- Best model: DoRA r=64"
echo "- Fastest: LoRA r=32"
echo "- Minimum dataset: 20 images"
echo ""
echo "Next steps:"
echo "1. Review training curves in $BASE_OUTPUT/"
echo "2. Check generated images in $BASE_OUTPUT/generated_*/"
echo "3. Launch demo: python gradio_demo.py --models $BASE_OUTPUT/lora_r32 $BASE_OUTPUT/dora_r64"
echo ""
echo "=================================================="
