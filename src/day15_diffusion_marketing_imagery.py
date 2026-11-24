"""RetailMind AI - Day 15: Stable Diffusion marketing imagery.

Library focus: Hugging Face diffusers.
No training dataset is required. Prompts are built from local product metadata.
Default pretrained model: runwayml/stable-diffusion-v1-5.

Expected local metadata file: data/product_marketing_metadata.json
"""

from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
METADATA_PATH = "data/product_marketing_metadata.json"
OUTPUT_DIR = "generated/day15_marketing"
PROMPT_LIBRARY_PATH = "generated/day15_marketing/prompt_library.json"
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, duplicate product, extra objects, "
    "cropped product, watermark, logo distortion, text, typography, oversaturated"
)
DEFAULT_STYLE = "premium modern retail campaign, clean composition, realistic product photography"
DEFAULT_GUIDANCE = 7.5
DEFAULT_STEPS = 30
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def load_metadata(path=METADATA_PATH):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    products = data.get("products", data if isinstance(data, list) else [])
    if not products:
        raise ValueError("Metadata must contain a non-empty products list.")
    required = {"product_id", "category", "product_name"}
    for item in products:
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Product metadata missing: {sorted(missing)}")
    return products


def build_prompt(product, campaign="hero"):
    style = product.get("brand_style", DEFAULT_STYLE)
    setting = product.get("setting", "minimal studio background")
    colors = product.get("brand_colors", "balanced neutral palette")
    material = product.get("material", "realistic commercial product finish")
    campaign_text = {
        "hero": "single hero product, centered commercial composition",
        "social": "social media campaign composition, dynamic but uncluttered",
        "catalog": "clean e-commerce catalog composition, product fully visible",
    }.get(campaign, campaign)
    return (
        f"Professional advertising photograph of {product['product_name']}, "
        f"category {product['category']}, {material}, {campaign_text}, {setting}, "
        f"brand palette {colors}, {style}, photorealistic, studio lighting, high detail"
    )


def build_negative_prompt(product):
    extra = product.get("negative_prompt", "")
    return ", ".join(part for part in [DEFAULT_NEGATIVE, extra] if part)


def load_pipeline(model_id=MODEL_ID, local_files_only=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Day 15 Stable Diffusion generation requires a CUDA GPU runtime.")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=local_files_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def generate_asset(pipe, product, seed=42, campaign="hero", guidance_scale=DEFAULT_GUIDANCE):
    prompt = build_prompt(product, campaign)
    negative = build_negative_prompt(product)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        guidance_scale=guidance_scale,
        num_inference_steps=DEFAULT_STEPS,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        generator=generator,
    )
    return result.images[0], {
        "product_id": product["product_id"],
        "campaign": campaign,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "steps": DEFAULT_STEPS,
        "prompt": prompt,
        "negative_prompt": negative,
        "model": MODEL_ID,
    }


def quality_review(asset):
    """Return a human-review checklist. Diffusion output still needs brand/product approval."""
    return {
        "product_identity_preserved": None,
        "brand_colors_consistent": None,
        "no_unwanted_text_or_watermark": None,
        "no_distorted_geometry": None,
        "product_fully_visible": None,
        "campaign_appropriate": None,
        "approved": None,
        "review_notes": "",
    }


def save_asset(image, metadata, output_dir=OUTPUT_DIR):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{metadata['product_id']}_{metadata['campaign']}_seed{metadata['seed']}"
    image_path = output / f"{stem}.png"
    json_path = output / f"{stem}.json"
    image.save(image_path)
    payload = {**metadata, "quality_review": quality_review(metadata)}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return image_path, json_path


def build_prompt_library(products):
    library = []
    for product in products:
        for campaign in ("hero", "social", "catalog"):
            for seed in (42, 108, 2026):
                library.append({
                    "product_id": product["product_id"],
                    "campaign": campaign,
                    "seed": seed,
                    "prompt": build_prompt(product, campaign),
                    "negative_prompt": build_negative_prompt(product),
                    "guidance_scale": DEFAULT_GUIDANCE,
                })
    return library


def main():
    products = load_metadata()
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    library = build_prompt_library(products)
    Path(PROMPT_LIBRARY_PATH).write_text(json.dumps(library, indent=2), encoding="utf-8")
    print("Saved reproducible prompt library:", PROMPT_LIBRARY_PATH)

    pipe = load_pipeline()
    for product in products:
        image, metadata = generate_asset(pipe, product, seed=42, campaign="hero")
        image_path, metadata_path = save_asset(image, metadata)
        print("Generated:", image_path)
        print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
