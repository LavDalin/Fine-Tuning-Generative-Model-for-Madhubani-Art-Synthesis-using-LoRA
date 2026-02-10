# Quick Start Guide - Madhubani Art Fine-Tuning

## Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
# Option A: Automatic setup
chmod +x setup.sh
./setup.sh

# Option B: Manual setup
conda create -n madhubani python=3.10 -y
conda activate madhubani
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset (1 minute)

1. Create directory structure:
```bash
mkdir -p madhubani_dataset
```

2. Add your images to `madhubani_dataset/`:
   - At least 10 images (20+ recommended, 50 optimal)
   - JPG or PNG format
   - 512x512 or higher resolution

3. Create caption files:
   - For each `image.jpg`, create `image.txt`
   - Use this template:
   ```
   madhubani art painting, [describe subject], intricate geometric patterns, 
   double-line borders, vibrant red blue yellow green colors, traditional Indian folk art
   ```

**Example captions:**
```
# elephant.txt
madhubani art painting, sacred elephant adorned with traditional jewelry and 
decorative patterns, bold outlines, vibrant red blue yellow green colors, folk art motifs

# peacock.txt
madhubani art painting, majestic peacock with ornate tail feathers, intricate 
geometric patterns, bold black outlines, vibrant colors, traditional indian folk art
```

### Step 3: Train Your First Model (2-3 minutes)

```bash
# Quick training with LoRA (fastest, 2.2 min)
python train_lora.py \
    --config_name lora_r32 \
    --dataset_path ./madhubani_dataset \
    --output_dir ./my_first_model
```

### Step 4: Generate Images

```bash
python inference.py \
    --model_path ./my_first_model \
    --prompt "madhubani art painting, sacred elephant with decorative patterns" \
    --num_images 4
```

That's it! Check `generated_images/` for your results.

---

## Next Steps

### Train Better Models

**For best quality (DoRA r=64):**
```bash
python train_lora.py \
    --config_name dora_r64 \
    --dataset_path ./madhubani_dataset \
    --output_dir ./best_model
```

**With tracking (W&B):**
```bash
python train_lora.py \
    --config_name dora_r64 \
    --dataset_path ./madhubani_dataset \
    --use_wandb \
    --wandb_project my-madhubani-project
```

### Compare Models

```bash
# Launch interactive web interface
python gradio_demo.py \
    --models ./my_first_model ./best_model \
    --names "LoRA r=32" "DoRA r=64" \
    --share
```

### Run All Experiments

```bash
# Replicate the paper's experiments
chmod +x run_experiments.sh
./run_experiments.sh
```

---

## Generation Tips

### Best Prompts
Always include:
- "madhubani art painting" (trigger phrase)
- Subject description
- "intricate geometric patterns"
- "double-line borders"
- Color names
- "traditional Indian folk art"

### Best Settings
- **Steps**: 25-30
- **Guidance Scale**: 7.0-7.5
- **LoRA Scale**: 1.3-1.5 (higher = stronger style)
- **Scheduler**: DPM++ 2M Karras

### Example Command
```bash
python inference.py \
    --model_path ./best_model \
    --prompt "madhubani art painting, sacred Ganesh with ornate decorations, 
              intricate geometric patterns, double-line borders, vibrant red 
              yellow blue green colors, traditional folk art motifs" \
    --num_images 4 \
    --lora_scale 1.4 \
    --scheduler dpm++ \
    --guidance_scale 7.5
```

---

## Troubleshooting

### Out of Memory?
```python
# In config.py, reduce batch size:
config.train_batch_size = 2
config.gradient_accumulation_steps = 8
```

### Poor Results?
1. Increase LoRA scale: `--lora_scale 1.5`
2. Train longer: Use 40 epochs in config
3. Use more images: Aim for 50
4. Try DoRA: `--config_name dora_r64`

### Slow Training?
```bash
# Use smaller rank
python train_lora.py --config_name lora_r16 ...

# Install xformers for faster attention
pip install xformers
```

---

## File Structure

```
madhubani-art-finetuning/
├── train_lora.py          # Main training script
├── inference.py           # Generate images
├── gradio_demo.py        # Web interface
├── config.py             # All configurations
├── dataset.py            # Dataset handling
├── evaluation.py         # Analysis tools
├── utils.py              # Utilities
├── examples.py           # Usage examples
├── setup.sh              # Installation script
├── run_experiments.sh    # Full experiments
├── requirements.txt      # Dependencies
├── README.md             # Full documentation
└── QUICKSTART.md         # This file
```

---

## Common Commands Cheat Sheet

```bash
# Basic training
python train_lora.py --config_name lora_r32 --dataset_path ./madhubani_dataset

# Best quality training
python train_lora.py --config_name dora_r64 --dataset_path ./madhubani_dataset

# Training with subset
python train_lora.py --config_name lora_r32 --dataset_path ./madhubani_dataset --subset_size 20

# Generate images
python inference.py --model_path ./output --prompt "your prompt here"

# Batch generate
for prompt in "elephant" "peacock" "ganesh"; do
    python inference.py --model_path ./output --prompt "madhubani art painting, $prompt"
done

# Launch demo
python gradio_demo.py --models ./model1 ./model2 --share

# Run examples
python examples.py

# Validate dataset
python utils.py
```

---

## Learn More

- **Full Documentation**: See `README.md`
- **Example Code**: Run `python examples.py`
- **Configuration Guide**: See `config.py`
- **Paper Results**: Run `./run_experiments.sh`

---

## Need Help?

1. Check `README.md` for detailed docs
2. Run `python examples.py` for code samples
3. See troubleshooting section above
4. Review the paper's methodology

Happy generating!
