# Day 15 - Stable Diffusion Marketing Imagery

## Purpose
Day 15 adds controlled text-to-image generation to RetailMind AI. The module uses a pretrained Stable Diffusion pipeline to create draft hero, social, and catalog campaign imagery from approved product metadata.

## Data source
There is no new training dataset. The pretrained model is `runwayml/stable-diffusion-v1-5`. RetailMind provides local metadata only:

```text
product_id
product_name
category
material
brand_colors
brand_style
setting
negative_prompt (optional)
```

Example file: `data/product_marketing_metadata.json`.

## Prompt engineering
The prompt template combines product name, category, material, campaign composition, setting, brand palette and brand style. This creates a repeatable contract between product metadata and creative generation rather than allowing uncontrolled free-form prompts.

Three default campaign modes are provided:

- `hero`: centered single-product advertising composition;
- `social`: more dynamic social-campaign composition;
- `catalog`: clean e-commerce composition with the product fully visible.

## Negative prompt
The default negative prompt rejects common failure modes including blur, poor quality, deformed objects, duplicate products, cropping, watermarks, unwanted text, typography and oversaturation. Product-specific negative terms may be appended from metadata.

Negative prompts reduce unwanted patterns but do not guarantee correctness. Human review remains mandatory.

## Reproducibility
Every approved candidate should preserve:

- pretrained model identifier;
- full positive prompt;
- full negative prompt;
- campaign mode;
- random seed;
- guidance scale;
- scheduler/inference-step count;
- image dimensions.

The default library creates three campaign styles × three fixed seeds per product. This makes creative comparisons repeatable.

## Generation defaults

```text
Model: runwayml/stable-diffusion-v1-5
Resolution: 512 x 512
Guidance scale: 7.5
Inference steps: 30
Scheduler: DPMSolverMultistepScheduler
Precision: float16
Runtime: CUDA GPU
```

These values are practical starting points and should be validated against the available GPU and brand-quality requirements.

## Quality-control checklist
An asset is not approved solely because it looks attractive. Review:

1. product identity preservation;
2. geometry and component count;
3. brand-color consistency;
4. material realism;
5. accidental text/logos/watermarks;
6. product visibility and cropping;
7. campaign appropriateness;
8. potentially misleading visual claims.

Each saved asset receives a JSON quality-review block to support manual approval notes.

## Outputs

```text
generated/day15_marketing/prompt_library.json
generated/day15_marketing/<product>_<campaign>_seed<seed>.png
generated/day15_marketing/<product>_<campaign>_seed<seed>.json
```

The image and JSON should remain paired for auditability.

## Run

```bash
python3 src/day15_diffusion_marketing_imagery.py
```

Generation requires a CUDA GPU and access to the pretrained weights from Hugging Face or a valid local cache.

## Risks and limitations
- Text-to-image generation may change exact SKU shape or physical details.
- Generated brand marks and text are unreliable.
- Higher guidance is not always higher quality and can reduce naturalness.
- A fixed seed aids reproducibility but does not make output objectively correct.
- Model-license and organizational usage requirements must be checked before commercial deployment.
- Generated marketing concepts should never silently replace authoritative product photography.

## Day 16 handoff
Day 16 can preserve more product identity by starting from an approved real or generated image with img2img, then create controlled variations and a short promotional clip.
