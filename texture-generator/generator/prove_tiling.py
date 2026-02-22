"""
Prove tileable texture generation with SDXL Base + circular padding.

Generates a 1024x1024 tile, a 2x2 grid (2048x2048), and a 4x4 grid (4096x4096)
so you can visually verify there are no seams.

Usage:
    python3 prove_tiling.py "red brick wall"
    python3 prove_tiling.py "fire"
    python3 prove_tiling.py "mossy stone" --steps 40
"""

import argparse
import logging

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_conv2d_circular(module: nn.Module) -> int:
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
            m.padding_mode = "circular"
            count += 1
    return count


def make_grid(tile: Image.Image, repeats: int = 2) -> Image.Image:
    w, h = tile.size
    grid = Image.new("RGB", (w * repeats, h * repeats))
    for row in range(repeats):
        for col in range(repeats):
            grid.paste(tile, (col * w, row * h))
    return grid


def main():
    parser = argparse.ArgumentParser(description="Prove SDXL tileable textures")
    parser.add_argument("prompt", nargs="*", default=["fire"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=7.0)
    args = parser.parse_args()

    prompt = " ".join(args.prompt)
    safe_name = prompt.replace(" ", "_").lower()[:40]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Device: {device}, dtype: {dtype}")

    logger.info("Loading SDXL Base 1.0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
    )

    logger.info("Patching Conv2d layers for circular padding...")
    unet_count = patch_conv2d_circular(pipe.unet)
    vae_count = patch_conv2d_circular(pipe.vae)
    logger.info(f"Patched {unet_count} UNet + {vae_count} VAE Conv2d layers")

    pipe.enable_attention_slicing()
    pipe.to(device)

    enhanced = (
        f"seamless tileable texture pattern of {prompt}, "
        "top-down orthographic view, even flat lighting, no shadows, "
        "highly detailed, sharp focus, 8k, PBR material, "
        "uniform density, perfectly repeating seamless pattern"
    )
    negative = (
        "blurry, low quality, watermark, text, logo, border, frame, "
        "vignette, perspective, 3d render, human, face, "
        "noisy, jpeg artifacts, visible seam, uneven lighting"
    )

    generator = torch.Generator(device="cpu")

    logger.info(f"Generating: '{prompt}' at {args.size}x{args.size}, {args.steps} steps, cfg={args.guidance}")

    result = pipe(
        prompt=enhanced,
        negative_prompt=negative,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )

    tile = result.images[0]
    grid_2x2 = make_grid(tile, 2)
    grid_4x4 = make_grid(tile, 4)

    tile_path = f"{safe_name}_tile.png"
    grid_2x2_path = f"{safe_name}_tiled_2x2.png"
    grid_4x4_path = f"{safe_name}_tiled_4x4.png"

    tile.save(tile_path)
    grid_2x2.save(grid_2x2_path)
    grid_4x4.save(grid_4x4_path)

    logger.info(f"Saved: {tile_path} ({args.size}x{args.size})")
    logger.info(f"Saved: {grid_2x2_path} ({args.size*2}x{args.size*2})")
    logger.info(f"Saved: {grid_4x4_path} ({args.size*4}x{args.size*4})")
    logger.info("Check the grids for seams at tile boundaries.")


if __name__ == "__main__":
    main()
