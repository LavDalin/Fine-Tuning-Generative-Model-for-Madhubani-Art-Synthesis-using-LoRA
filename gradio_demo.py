import gradio as gr
import torch
import torch.nn as nn
import os
import types
import time
import gc
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# HyperLoRA Architecture (must match training)
# ---------------------------------------------------------------------------

@dataclass
class HyperLoRAConfig:
    rank: int = 8
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    hypernetwork_hidden_size: int = 256
    hypernetwork_num_layers: int = 2
    alpha: float = 1.0


class HyperNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2,
                 lora_rank=8, target_dim_a=768, target_dim_b=768):
        super().__init__()
        self.lora_rank = lora_rank
        self.target_dim_a = target_dim_a
        self.target_dim_b = target_dim_b
        layers, current_dim = [], input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)])
            current_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)
        self.output_A = nn.Linear(hidden_dim, lora_rank * target_dim_a)
        self.output_B = nn.Linear(hidden_dim, target_dim_b * lora_rank)

    def forward(self, context):
        hidden = self.shared_net(context)
        A = self.output_A(hidden).view(-1, self.lora_rank, self.target_dim_a)
        B = self.output_B(hidden).view(-1, self.target_dim_b, self.lora_rank)
        return A, B


class HyperLoRALayer(nn.Module):
    def __init__(self, original_layer, hypernetwork, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.hypernetwork = hypernetwork
        self.alpha = alpha

    def forward(self, x, context=None):
        result = self.original_layer(x)
        if context is not None:
            if context.dim() > 2:
                context = context.mean(dim=1)
            A, B = self.hypernetwork(context)
            result = result + self.alpha * torch.einsum('bi,brj,bjr->bi', x, A, B)
        return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MadhubaniGenerator:
    def __init__(self):
        self.pipe = None
        self.hypernets = None
        self.current_lora = None
        self.base_model_id = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_configs = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scheduler(self, config):
        return DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )

    def _patch_unet_for_hyperlora(self, pipe):
        orig = pipe.unet.forward
        def patched(self_unet, sample, timestep, encoder_hidden_states, context=None, **kw):
            return orig(sample, timestep, encoder_hidden_states, **kw)
        pipe.unet.forward = types.MethodType(patched, pipe.unet)

    def _load_base_pipeline(self) -> StableDiffusionPipeline:
        print(f"  Loading base SD 1.5 from HuggingFace...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_id, torch_dtype=self.dtype, safety_checker=None
        ).to(self.device)
        pipe.scheduler = self._scheduler(pipe.scheduler.config)
        return pipe

    def _inject_hyperlora(self, pipe, config: HyperLoRAConfig) -> nn.ModuleDict:
        hypernets = nn.ModuleDict()
        for name, module in pipe.unet.named_modules():
            if any(t in name for t in config.target_modules) and isinstance(module, nn.Linear):
                hnet = HyperNetwork(lora_rank=config.rank,
                                    target_dim_a=module.in_features,
                                    target_dim_b=module.out_features)
                hypernets[name.replace(".", "_")] = hnet
                parent_name, child_name = (name.rsplit('.', 1) if '.' in name else ('', name))
                parent = pipe.unet.get_submodule(parent_name) if parent_name else pipe.unet
                setattr(parent, child_name, HyperLoRALayer(module, hnet, alpha=config.alpha))
        return hypernets

    def _release_current(self):
        """Delete pipeline without moving to CPU (float16 + CPU = error)."""
        if self.pipe is not None:
            print("  Releasing pipeline from GPU...")
            del self.pipe
            self.pipe = None
        if self.hypernets is not None:
            del self.hypernets
            self.hypernets = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_lora_weight_file(self, lora_dir: str) -> Optional[str]:
        """Return path to the first recognised LoRA weight file in lora_dir."""
        for fname in [
            "pytorch_lora_weights.safetensors",
            "adapter_model.safetensors",
            "pytorch_lora_weights.bin",
            "adapter_model.bin",
        ]:
            p = os.path.join(lora_dir, fname)
            if os.path.isfile(p):
                return p
        # Last resort: any .safetensors or .bin
        for fname in os.listdir(lora_dir):
            if fname.endswith(".safetensors") or fname.endswith(".bin"):
                return os.path.join(lora_dir, fname)
        return None

    def _load_lora_pipeline(self, output_dir: str) -> StableDiffusionPipeline:
        """
        Your training saves a full pipeline EXCEPT the UNet (only the LoRA delta
        is stored in lora_weights/).  Strategy:
          1. Load base SD 1.5 (gets us the UNet).
          2. Override scheduler/VAE/text-encoder/tokenizer from the saved output
             dir if they exist there (they may have been fine-tuned too).
          3. Inject the LoRA weights from lora_weights/ using load_lora_weights
             with prefix=None so diffusers doesn't filter out unrecognised keys.
        """
        # Step 1 – base pipeline
        pipe = self._load_base_pipeline()

        # Step 2 – swap components that were saved locally
        component_loaders = {
            "scheduler":     lambda: DPMSolverMultistepScheduler.from_pretrained(
                                 output_dir, subfolder="scheduler"),
            "tokenizer":     lambda: CLIPTokenizer.from_pretrained(
                                 output_dir, subfolder="tokenizer"),
            "text_encoder":  lambda: CLIPTextModel.from_pretrained(
                                 output_dir, subfolder="text_encoder",
                                 torch_dtype=self.dtype).to(self.device),
        }
        for comp_name, loader in component_loaders.items():
            comp_dir = os.path.join(output_dir, comp_name)
            if os.path.isdir(comp_dir):
                try:
                    loaded = loader()
                    setattr(pipe, comp_name, loaded)
                    print(f"    Swapped {comp_name} from saved checkpoint.")
                except Exception as e:
                    print(f"    Could not swap {comp_name}: {e}")

        # Step 3 – inject LoRA delta
        lora_dir = os.path.join(output_dir, "lora_weights")
        if not os.path.isdir(lora_dir):
            print(f"  WARNING: No lora_weights/ subdir found in {output_dir}. Using base UNet.")
            return pipe

        weight_file = self._find_lora_weight_file(lora_dir)
        if weight_file is None:
            print(f"  WARNING: No weight file found in {lora_dir}. Using base UNet.")
            return pipe

        print(f"  Injecting LoRA from: {weight_file}")

        # Inspect key prefixes so we can set prefix= correctly
        try:
            from safetensors import safe_open
            with safe_open(weight_file, framework="pt") as f:
                sample_keys = list(f.keys())[:8]
            print(f"    Key sample: {sample_keys}")
        except Exception:
            sample_keys = []

        # If keys already carry unet./text_encoder. prefix → use default (no kwarg needed)
        # If keys use bare lora_A/lora_B style → pass prefix=None
        has_module_prefix = any(
            k.startswith("unet.") or k.startswith("text_encoder.") for k in sample_keys
        )
        prefix_kwarg = {} if has_module_prefix else {"prefix": None}

        adapter_name = "lora_adapter"
        try:
            pipe.load_lora_weights(lora_dir, adapter_name=adapter_name, **prefix_kwarg)
            # Verify registration
            registered = set(getattr(pipe.unet, "peft_config", {}).keys())
            registered |= set(getattr(pipe.text_encoder, "peft_config", {}).keys())
            if adapter_name in registered:
                pipe.set_adapters(adapter_name)
                print(f"  LoRA adapter activated successfully.")
            elif registered:
                fallback = next(iter(registered))
                pipe.set_adapters(fallback)
                print(f"  Activated adapter '{fallback}' (registered under different name).")
            else:
                # PEFT registration failed — try manual state-dict loading as last resort
                print("  PEFT registration failed. Attempting manual weight merge...")
                pipe = self._manual_lora_merge(pipe, weight_file)
        except Exception as e:
            print(f"  load_lora_weights error: {e}. Attempting manual weight merge...")
            pipe = self._manual_lora_merge(pipe, weight_file)

        return pipe

    def _manual_lora_merge(self, pipe, weight_file: str) -> StableDiffusionPipeline:
        """
        Last-resort: load the raw A/B matrices and add lora_scale * B @ A to the
        matching Linear weight in the UNet.  Works regardless of key naming scheme.
        """
        try:
            from safetensors.torch import load_file
            state = load_file(weight_file, device="cpu")
        except Exception:
            state = torch.load(weight_file, map_location="cpu", weights_only=True)

        print(f"  Manual merge: {len(state)} tensors loaded.")

        # Build a map of {bare_module_name: {lora_A: tensor, lora_B: tensor}}
        lora_map = {}
        for k, v in state.items():
            # Strip common prefixes: unet., text_encoder., base_model.model., etc.
            bare = k
            for prefix in ["unet.", "text_encoder.", "base_model.model.", "model."]:
                if bare.startswith(prefix):
                    bare = bare[len(prefix):]
                    break
            # Identify A/B
            if "lora_A" in bare or ".lora_down" in bare:
                module_key = bare.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
                lora_map.setdefault(module_key, {})["A"] = v.float()
            elif "lora_B" in bare or ".lora_up" in bare:
                module_key = bare.replace(".lora_B.weight", "").replace(".lora_up.weight", "")
                lora_map.setdefault(module_key, {})["B"] = v.float()

        merged = 0
        for module_key, mats in lora_map.items():
            if "A" not in mats or "B" not in mats:
                continue
            # Find matching module in UNet
            try:
                # Convert dot-path: attn1.to_q → attn1.to_q
                module = pipe.unet.get_submodule(module_key)
                if isinstance(module, nn.Linear):
                    delta = (mats["B"] @ mats["A"]).to(self.dtype).to(self.device)
                    with torch.no_grad():
                        module.weight.data += delta
                    merged += 1
            except Exception:
                pass

        print(f"  Manual merge complete: {merged} modules updated.")
        return pipe

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def switch_model(self, model_name: str):
        if self.current_lora == model_name:
            return

        self._release_current()

        path = self.model_configs.get(model_name, "BASE")

        # ── Base SD v1.5 ──────────────────────────────────────────────
        if path == "BASE":
            print("Loading Base SD v1.5...")
            self.pipe = self._load_base_pipeline()
            self.current_lora = model_name
            print("  Ready.")
            return

        # ── HyperLoRA ────────────────────────────────────────────────
        if "hyperlora" in path.lower():
            print(f"Loading HyperLoRA: {model_name}")
            self.pipe = self._load_base_pipeline()
            self._patch_unet_for_hyperlora(self.pipe)
            self.hypernets = self._inject_hyperlora(self.pipe, HyperLoRAConfig(rank=8))
            weights = torch.load(
                os.path.join(path, "hyperlora_weights.pt"),
                map_location=self.device,
                weights_only=True,
            )
            self.hypernets.load_state_dict(weights)
            self.hypernets.to(self.device, dtype=self.dtype)
            self.current_lora = model_name
            print("  HyperLoRA ready.")
            return

        # ── LoRA / DoRA checkpoint ────────────────────────────────────
        print(f"Loading LoRA/DoRA model: {model_name}")
        self.pipe = self._load_lora_pipeline(path)
        self.current_lora = model_name
        print(f"  Ready: {model_name}")

    def generate(self, model_name, prompt, negative_prompt, steps, guidance, lora_scale, seed):
        self.switch_model(model_name)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        path = self.model_configs.get(model_name, "")
        extra_kwargs = {}
        if self.hypernets is not None and "hyperlora" in (path or "").lower():
            inputs = self.pipe.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                extra_kwargs["context"] = self.pipe.text_encoder(inputs.input_ids)[0]

        registered = set(getattr(self.pipe.unet, "peft_config", {}).keys())
        cross_attn = {"scale": lora_scale} if registered else {}

        start = time.time()
        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                cross_attention_kwargs=cross_attn,
                **extra_kwargs,
            ).images[0]
            return image, f"✅ {time.time()-start:.2f}s  |  {self.current_lora}"
        except Exception as e:
            return None, f"❌ Error: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

generator_instance = MadhubaniGenerator()

EXAMPLE_PROMPTS = {
    "Elephant": "madhubani art painting, sacred elephant adorned with traditional jewelry and decorative patterns, bold outlines, vibrant colors",
    "Peacock":  "madhubani art painting, majestic peacock with ornate tail feathers, intricate geometric patterns, traditional folk art",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--names",  nargs='+')
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args()

    model_map = {"Base SD v1.5": "BASE"}
    if args.names and len(args.names) == len(args.models):
        for name, path in zip(args.names, args.models):
            model_map[name] = path
    else:
        for p in args.models:
            model_map[Path(p).name] = p

    generator_instance.model_configs = model_map

    with gr.Blocks(title="Madhubani Art Comparison") as demo:
        gr.Markdown("# 🎨 Madhubani Art Synthesis — Model Comparison")
        with gr.Row():
            with gr.Column():
                m_sel    = gr.Dropdown(list(model_map.keys()), label="Model", value="Base SD v1.5")
                p_in     = gr.Textbox(label="Prompt", value=EXAMPLE_PROMPTS["Elephant"])
                np_in    = gr.Textbox(label="Negative Prompt", value="blurry, low quality, ugly, deformed")
                btn      = gr.Button("Generate", variant="primary")
                with gr.Row():
                    steps    = gr.Slider(10, 50,  value=30,  step=1,    label="Steps")
                    guidance = gr.Slider(1,  15,  value=7.5, step=0.5,  label="Guidance")
                    scale    = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="LoRA Scale")
                seed = gr.Number(value=42, label="Seed", precision=0)
            with gr.Column():
                out_img = gr.Image(label="Output")
                out_txt = gr.Textbox(label="Info", interactive=False)

        btn.click(
            fn=generator_instance.generate,
            inputs=[m_sel, p_in, np_in, steps, guidance, scale, seed],
            outputs=[out_img, out_txt],
        )

    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
