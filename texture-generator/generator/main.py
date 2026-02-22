"""
Texture Generator Service v3.0

High-quality tileable texture generation using SDXL Base 1.0 (~7GB).
Seamless tiling via circular padding on all Conv2d layers (torus topology).

Upgrade from v2.0 (SDXL + SDXL-Lightning):
- Removed Lightning LoRA (was crippling prompt following at 1.5 guidance)
- Proper inference: 30 steps, guidance 7.0 for faithful prompt adherence
- Circular padding on both UNet + VAE for guaranteed seamless tiling
- Material-aware prompt engineering
- Sobel-based displacement with circular boundary conditions
- Normal map generation endpoint
"""

import io
import logging

import numpy as np
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Texture Generator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None
device = None

# ── Material-aware prompt enhancement ──────────────────────────────────────

MATERIAL_HINTS = {
    "wood": "natural wood grain, timber surface, annual rings",
    "metal": "metallic surface, brushed metal finish, alloy",
    "stone": "natural stone surface, geological detail, mineral",
    "brick": "brickwork, mortar joints, clay brick masonry",
    "fabric": "woven textile, fabric weave pattern, thread detail",
    "water": "water surface, ripples, aquatic caustics, reflections",
    "fire": "flames, ember glow, combustion, heat distortion",
    "concrete": "poured concrete, cement surface, aggregate",
    "sand": "granular sandy surface, beach sand, fine particles",
    "leather": "leather surface, tanned hide, pores and creases",
    "marble": "marble surface, veined stone, polished mineral",
    "grass": "grass blades, lawn surface, green vegetation",
    "ice": "frozen ice surface, crystalline frost, frozen",
    "rust": "corroded metal, oxidized iron, rust patina, decay",
    "lava": "molten lava, volcanic surface, magma flow, incandescent",
    "moss": "moss-covered surface, bryophyte, green growth",
    "bark": "tree bark, wooden trunk, cortex texture, rough surface",
    "scales": "scaled surface, reptilian, overlapping scales",
    "crystal": "crystalline formation, faceted mineral, prismatic",
    "fur": "animal fur, hair strands, soft pelage",
    "rock": "rock face, geological formation, rough mineral",
    "tile": "ceramic tile surface, glazed finish, grout lines",
    "paint": "painted surface, brush strokes, pigment layers",
    "cloth": "woven cloth, textile weave, draped fabric fibers",
    "dirt": "soil surface, earth ground, terrain, organic matter",
    "mud": "wet mud surface, clay earth, viscous",
    "snow": "snow surface, powder snow, frozen crystals",
    "cloud": "cloud formation, atmospheric, volumetric vapor",
    "smoke": "smoke wisps, vapor trails, billowing",
    "energy": "energy field, plasma, electrical discharge",
    "circuit": "circuit board, electronic traces, PCB, chips",
    "neon": "neon glow, luminescent, fluorescent emission",
    "galaxy": "galactic nebula, cosmic gas, stellar dust",
    "abstract": "abstract pattern, geometric forms, procedural",
    "glass": "glass surface, refractive, translucent, smooth",
    "coral": "coral reef surface, marine organism, porous",
    "bone": "bone surface, osseous tissue, calcified",
    "chain": "chainmail, interlocking metal rings, armor",
    "hex": "hexagonal pattern, honeycomb, tessellation",
    "wave": "wave pattern, undulating surface, rhythmic",
    "fractal": "fractal pattern, recursive detail, self-similar",
    "slime": "viscous slime, glossy organic, translucent goo",
}


def enhance_prompt(prompt: str) -> tuple[str, str]:
    """Build SDXL-optimized prompt pair with material-type awareness."""
    prompt_lower = prompt.lower()
    hints = [hint for key, hint in MATERIAL_HINTS.items() if key in prompt_lower]
    hint_str = f", {', '.join(hints)}" if hints else ""

    enhanced = (
        f"seamless tileable texture pattern of {prompt}{hint_str}, "
        "top-down orthographic view, even flat lighting, no directional shadows, "
        "highly detailed, sharp focus, 8k resolution, PBR material texture, "
        "uniform density across entire image, perfectly repeating seamless pattern, "
        "professional game texture asset, substance designer quality"
    )

    negative = (
        "blurry, soft focus, low quality, low resolution, watermark, text, logo, "
        "border, frame, vignette, perspective distortion, vanishing point, "
        "3d render, photograph, depth of field, bokeh, "
        "human, person, face, fingers, animal, "
        "noisy, jpeg artifacts, compression artifacts, color banding, "
        "visible seam, edge discontinuity, uneven lighting, spotlight, gradient"
    )

    return enhanced, negative


# ── Circular padding for seamless tiling ───────────────────────────────────


def patch_conv2d_circular(module: nn.Module) -> int:
    """Patch all Conv2d layers to use circular padding for seamless tiling.

    Forces the diffusion process to operate on a torus topology,
    guaranteeing that opposite edges of the generated image are continuous.
    Applied to both UNet and VAE for full coverage.
    """
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
            m.padding_mode = "circular"
            count += 1
    return count


# ── Pipeline loading ───────────────────────────────────────────────────────


def load_pipeline():
    """Load SDXL Base 1.0 with circular padding on UNet + VAE."""
    global pipe, device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS + float16 produces NaN in SDXL; use float32 for stability
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Using device: {device}, dtype: {dtype}")

    logger.info("Loading SDXL Base 1.0 (~7GB fp16, ~14GB fp32)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
    )

    # Circular padding for seamless tiling on both UNet and VAE
    logger.info("Patching Conv2d layers for circular padding...")
    unet_count = patch_conv2d_circular(pipe.unet)
    vae_count = patch_conv2d_circular(pipe.vae)
    logger.info(f"Patched {unet_count} UNet + {vae_count} VAE Conv2d layers")

    pipe.enable_attention_slicing()
    pipe.to(device)
    logger.info("Pipeline ready (SDXL Base + circular padding)")


@app.on_event("startup")
async def startup():
    load_pipeline()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "backend": "sdxl-base-1.0",
        "device": str(device) if device else "not loaded",
    }


# ── Texture generation ─────────────────────────────────────────────────────


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    guidance: float = Form(7.0),
):
    """
    Generate a tileable texture from a text prompt.

    - **prompt**: Text description (e.g. "red brick wall", "fire")
    - **width**: Output width, default 1024 (SDXL native), min 512, max 2048
    - **height**: Output height, default 1024
    - **steps**: Inference steps, default 30 (quality range: 20-50)
    - **guidance**: CFG scale, default 7.0 (range 5-12, higher = more prompt-faithful)
    """
    width = max(512, min(2048, (width // 8) * 8))
    height = max(512, min(2048, (height // 8) * 8))
    steps = max(10, min(50, steps))
    guidance = max(1.0, min(15.0, guidance))

    enhanced_prompt, negative_prompt = enhance_prompt(prompt)
    generator = torch.Generator(device="cpu")

    logger.info(
        f"Generating: '{prompt}' at {width}x{height}, steps={steps}, cfg={guidance}"
    )

    result = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ── PBR map generation ─────────────────────────────────────────────────────


def _sobel_circular(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients with circular (wrap-around) boundaries for seamless maps."""
    h = arr.astype(np.float64)
    padded = np.pad(h, 1, mode="wrap")

    # Sobel horizontal (dx)
    dx = (
        -1 * padded[:-2, :-2]
        + 1 * padded[:-2, 2:]
        + -2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        + -1 * padded[2:, :-2]
        + 1 * padded[2:, 2:]
    ) / 8.0

    # Sobel vertical (dy)
    dy = (
        -1 * padded[:-2, :-2]
        + -2 * padded[:-2, 1:-1]
        + -1 * padded[:-2, 2:]
        + 1 * padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + 1 * padded[2:, 2:]
    ) / 8.0

    return dx, dy


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance from RGB array (float64)."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


@app.post("/displacement")
async def displacement(image: UploadFile = File(...)):
    """
    Generate a height/displacement map from a color texture.

    Uses multi-scale luminance analysis with Sobel edge structure
    and circular boundary conditions for seamless results.
    """
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    arr = np.array(img, dtype=np.float64)

    gray = _luminance(arr)
    gray_img = Image.fromarray(gray.astype(np.uint8), mode="L")

    # Multi-scale decomposition
    base = np.array(
        gray_img.filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float64
    )
    fine = gray - np.array(
        gray_img.filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float64
    )

    # Edge structure via circular Sobel
    dx, dy = _sobel_circular(gray / 255.0)
    edge_mag = np.sqrt(dx**2 + dy**2)
    edge_mag = (edge_mag / (edge_mag.max() + 1e-9)) * 255

    # Blend: 60% smooth base + 25% fine detail + 15% edge structure
    combined = base * 0.60 + fine * 0.25 + edge_mag * 0.15
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
    combined = (combined * 255).clip(0, 255)

    result = Image.fromarray(combined.astype(np.uint8), mode="L")
    result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/normal")
async def normal_map(
    image: UploadFile = File(...),
    strength: float = Form(2.0),
):
    """
    Generate a tangent-space normal map from a color texture.

    Uses Sobel gradients with circular boundary conditions so the
    resulting normal map is itself seamlessly tileable.

    - **strength**: Normal intensity (default 2.0, range 0.5-10)
    """
    strength = max(0.5, min(10.0, strength))

    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    arr = np.array(img, dtype=np.float64)

    gray = _luminance(arr) / 255.0

    dx, dy = _sobel_circular(gray)
    dx *= strength
    dy *= strength

    # Tangent-space normal: R = X, G = Y, B = Z
    nx = -dx
    ny = -dy
    nz = np.ones_like(gray)

    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length

    normal_rgb = np.stack(
        [
            ((nx + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8),
            ((ny + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8),
            ((nz + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8),
        ],
        axis=-1,
    )

    result = Image.fromarray(normal_rgb, mode="RGB")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
