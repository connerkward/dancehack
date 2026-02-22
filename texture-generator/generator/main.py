"""
Texture Generator Service v3.1

High-quality tileable texture generation using SDXL Base 1.0 (~7GB).
Seamless tiling via circular padding on all Conv2d layers (torus topology).
MiDaS DPT_Large for high-quality displacement/depth maps.

Features:
- Circular padding on both UNet + VAE for guaranteed seamless tiling
- Material-aware prompt engineering
- MiDaS DPT_Large neural depth estimation for displacement maps
- Normal map generation endpoint
- IP-Adapter for reference image support
"""

import asyncio
import base64
import io
import json
import logging
import threading

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

app = FastAPI(title="Texture Generator", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None
device = None
gen_lock = threading.Lock()
midas_model = None
midas_transform = None

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
        f"close-up overhead photograph of {prompt} surface{hint_str}, "
        "top-down orthographic view, even flat lighting, "
        "highly detailed, sharp focus, 8k, "
        "no focal point, no center composition, "
        "random natural variation, irregular distribution, "
        "fills entire frame edge to edge uniformly"
    )

    negative = (
        "blurry, soft focus, low quality, low resolution, watermark, text, logo, "
        "border, frame, vignette, perspective distortion, vanishing point, "
        "3d render, depth of field, bokeh, "
        "human, person, face, fingers, animal, "
        "noisy, jpeg artifacts, compression artifacts, color banding, "
        "visible seam, uneven lighting, spotlight, gradient, "
        "obvious pattern, grid, symmetrical, repeating motif, centered object"
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

    # Load IP-Adapter for reference image support
    logger.info("Loading IP-Adapter for SDXL...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )
    pipe.set_ip_adapter_scale(0.0)
    logger.info("IP-Adapter loaded (default scale 0.0)")

    pipe.enable_attention_slicing()
    pipe.to(device)
    logger.info("Pipeline ready (SDXL Base + circular padding + IP-Adapter)")


def load_midas():
    """Load MiDaS DPT_Large for high-quality depth/displacement estimation (~1.2GB)."""
    global midas_model, midas_transform

    logger.info("Loading MiDaS DPT_Large (~1.2GB)...")
    midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas_model.eval()
    midas_model.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.dpt_transform
    logger.info("MiDaS DPT_Large ready")


@app.on_event("startup")
async def startup():
    load_pipeline()
    load_midas()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.1.0",
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


@app.post("/generate-stream")
async def generate_stream(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    guidance: float = Form(7.0),
    ip_adapter_scale: float = Form(0.0),
    reference_image: UploadFile | None = File(None),
):
    """Generate a tileable texture with SSE progress updates.

    Streams events:
      event: progress  data: {"step": N, "total": M}
      event: complete  data: {"image": "<base64 png>"}
      event: error     data: {"message": "..."}

    Optional IP-Adapter reference image:
      - reference_image: uploaded image file
      - ip_adapter_scale: strength 0-1 (0 = text only, 1 = image only)
    """
    width = max(512, min(2048, (width // 8) * 8))
    height = max(512, min(2048, (height // 8) * 8))
    steps = max(10, min(50, steps))
    guidance = max(1.0, min(15.0, guidance))
    ip_adapter_scale = max(0.0, min(1.0, ip_adapter_scale))

    # Read reference image before thread (UploadFile is async)
    ref_pil = None
    if reference_image is not None:
        try:
            ref_bytes = await reference_image.read()
            if ref_bytes:
                ref_pil = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
                logger.info(
                    f"Reference image loaded: {ref_pil.size}, "
                    f"ip_adapter_scale={ip_adapter_scale}"
                )
        except Exception as e:
            logger.warning(f"Failed to read reference image: {e}")

    progress = {"step": 0, "total": steps}
    image_holder: list = [None]
    error_holder: list = [None]
    done = threading.Event()

    def run():
        try:
            def on_step(p, step_index, timestep, cb_kwargs):
                progress["step"] = step_index + 1
                return cb_kwargs

            enhanced_prompt, negative_prompt = enhance_prompt(prompt)
            generator = torch.Generator(device="cpu")

            logger.info(
                f"Generating (stream): '{prompt}' at {width}x{height}, "
                f"steps={steps}, cfg={guidance}"
            )

            with gen_lock:
                # Set IP-Adapter scale (0.0 if no reference image)
                scale = ip_adapter_scale if ref_pil is not None else 0.0
                pipe.set_ip_adapter_scale(scale)

                kwargs = dict(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=on_step,
                )

                if ref_pil is not None and scale > 0:
                    kwargs["ip_adapter_image"] = ref_pil

                result = pipe(**kwargs)

                # Reset scale to 0 after generation
                pipe.set_ip_adapter_scale(0.0)

            image_holder[0] = result.images[0]
        except Exception as e:
            import traceback
            logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
            error_holder[0] = str(e)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()

    async def events():
        last_step = -1
        while not done.is_set():
            if progress["step"] != last_step:
                last_step = progress["step"]
                yield f"event: progress\ndata: {json.dumps(progress)}\n\n"
            await asyncio.sleep(0.2)

        # Emit final progress
        if progress["step"] != last_step:
            yield f"event: progress\ndata: {json.dumps(progress)}\n\n"

        if error_holder[0]:
            yield f"event: error\ndata: {json.dumps({'message': error_holder[0]})}\n\n"
        else:
            buf = io.BytesIO()
            image_holder[0].save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            yield f"event: complete\ndata: {json.dumps({'image': b64})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    Generate a height/displacement map from a color texture using MiDaS DPT_Large.

    Uses neural monocular depth estimation for realistic surface height inference,
    with seamless tiling post-processing via circular boundary blending.
    """
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    orig_w, orig_h = img.size

    # MiDaS transform expects numpy HWC uint8
    img_np = np.array(img)

    # Apply MiDaS transform and run inference
    input_batch = midas_transform(img_np).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)

        # Resize to original image dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to 0-255 range
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-9)

    # Invert: MiDaS outputs inverse depth (closer = higher values),
    # but for displacement we want raised surfaces = higher values
    depth = 1.0 - depth

    # Seamless tiling fix: blend edges with circular wrap
    # Use a feathered border to smooth any seam discontinuities
    border = max(8, min(orig_w, orig_h) // 32)
    depth_tiled = depth.copy()

    # Horizontal seam blending
    for i in range(border):
        alpha = i / border
        left_col = depth[:, i]
        right_col = depth[:, -(i + 1)]
        blended = left_col * (1 - alpha) + right_col * alpha
        depth_tiled[:, i] = depth[:, i] * alpha + blended * (1 - alpha)
        depth_tiled[:, -(i + 1)] = depth[:, -(i + 1)] * alpha + blended * (1 - alpha)

    # Vertical seam blending
    for i in range(border):
        alpha = i / border
        top_row = depth_tiled[i, :]
        bot_row = depth_tiled[-(i + 1), :]
        blended = top_row * (1 - alpha) + bot_row * alpha
        depth_tiled[i, :] = depth_tiled[i, :] * alpha + blended * (1 - alpha)
        depth_tiled[-(i + 1), :] = depth_tiled[-(i + 1), :] * alpha + blended * (1 - alpha)

    # Light Gaussian smooth to reduce any remaining micro-artifacts
    depth_uint8 = (depth_tiled * 255).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(depth_uint8, mode="L")
    result = result.filter(ImageFilter.GaussianBlur(radius=0.8))

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
