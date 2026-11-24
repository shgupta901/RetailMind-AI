"""RetailMind AI - Day 16: img2img variations and promo-frame generation.

Library focus: Hugging Face diffusers.
Input: a real catalog image from Fashion Product Images or another approved catalog source.
Output: controlled Stable Diffusion img2img variations plus an ordered frame sequence
that can be encoded into a short promo clip by an approved video tool.
"""

from pathlib import Path
import json
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
INIT_IMAGE = "data/day16_catalog_init.jpg"
OUTPUT_DIR = "generated/day16_variations"
MANIFEST_PATH = "generated/day16_variations/asset_manifest.json"
FRAME_DIR = "generated/day16_variations/promo_frames"
IMAGE_SIZE = 512
STEPS = 30
SEEDS = [42, 108, 2026]
STRENGTHS = [0.25, 0.45, 0.65]
GUIDANCE_SCALES = [6.0, 7.5, 9.0]

DEFAULT_PROMPT = (
    "premium retail campaign photograph of the same product, preserve product identity, "
    "clean studio composition, realistic materials, soft commercial lighting, high detail"
)
DEFAULT_NEGATIVE = (
    "different product, changed product shape, extra objects, duplicate product, deformed, "
    "blurry, low quality, cropped product, watermark, text, logo distortion"
)


def load_init_image(path=INIT_IMAGE):
    image = Image.open(path).convert("RGB")
    return image.resize((IMAGE_SIZE, IMAGE_SIZE))


def load_pipeline(model_id=MODEL_ID, local_files_only=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Day 16 img2img generation requires a CUDA GPU runtime.")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=local_files_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def generate_variation(pipe, init_image, prompt=DEFAULT_PROMPT, negative_prompt=DEFAULT_NEGATIVE,
                       strength=0.45, guidance_scale=7.5, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=STEPS,
        generator=generator,
    )
    return result.images[0]


def variation_grid(pipe, init_image, output_dir=OUTPUT_DIR):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for strength in STRENGTHS:
        for guidance in GUIDANCE_SCALES:
            seed = SEEDS[(STRENGTHS.index(strength) + GUIDANCE_SCALES.index(guidance)) % len(SEEDS)]
            image = generate_variation(
                pipe, init_image, strength=strength, guidance_scale=guidance, seed=seed
            )
            stem = f"variation_strength{strength:.2f}_guidance{guidance:.1f}_seed{seed}"
            path = output / f"{stem}.png"
            image.save(path)
            manifest.append({
                "file": str(path), "strength": strength, "guidance_scale": guidance,
                "seed": seed, "prompt": DEFAULT_PROMPT, "negative_prompt": DEFAULT_NEGATIVE,
                "model": MODEL_ID, "approved": None, "review_notes": ""
            })
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def interpolate_frames(image_a, image_b, frame_count=24, output_dir=FRAME_DIR):
    """Create ordered cross-fade frames between two approved campaign images.

    This uses PIL only to assemble the frames. Encoding PNG frames to MP4/GIF is intentionally
    left to the hosted environment's approved video encoder because diffusers generates the
    source imagery, not the container format.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    a = image_a.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    b = image_b.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    paths = []
    for i in range(frame_count):
        alpha = i / max(frame_count - 1, 1)
        frame = Image.blend(a, b, alpha)
        path = output / f"frame_{i:04d}.png"
        frame.save(path)
        paths.append(str(path))
    return paths


def curate_manifest(manifest, approved_files, output_path=MANIFEST_PATH):
    approved = set(approved_files)
    for item in manifest:
        item["approved"] = item["file"] in approved
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    init_image = load_init_image()
    pipe = load_pipeline()
    manifest = variation_grid(pipe, init_image)
    print("Generated variations:", len(manifest))
    print("Review manifest:", MANIFEST_PATH)
    print("After selecting two approved assets, call interpolate_frames() to create promo frames.")


if __name__ == "__main__":
    main()
