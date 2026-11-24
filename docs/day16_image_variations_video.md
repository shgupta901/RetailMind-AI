# Day 16 - Image Variations and Promo Video Frames

## Purpose
Day 16 extends RetailMind AI's generative marketing capability with image-to-image variation generation and a short promotional-frame workflow. The source product image anchors product identity while Stable Diffusion introduces controlled creative variation.

## Dataset
Day 16 uses a catalog product image as the initialization image. The intended source is the Fashion Product Images Small dataset or an approved RetailMind catalog photograph.

```text
data/day16_catalog_init.jpg
```

This differs from Day 15, where no image dataset was required and prompts were built only from metadata.

## Img2img architecture
The module uses `StableDiffusionImg2ImgPipeline` with Stable Diffusion v1.5. The original catalog image is encoded into the diffusion process and partially noised according to `strength`. The text prompt then guides denoising toward a campaign interpretation.

## Strength
Strength trades fidelity for creativity:

- `0.25`: preserve most product appearance;
- `0.45`: balanced campaign variation;
- `0.65`: stronger reinterpretation and higher identity risk.

Values are not quality scores. An apparently attractive high-strength image may no longer represent the real SKU.

## Guidance scale
The project tests `6.0`, `7.5`, and `9.0`. Higher guidance generally pushes stronger prompt adherence but can introduce unnatural or over-constrained images. Strength and guidance must be judged jointly.

## Reproducibility
Every candidate records:

- model ID;
- source-image contract;
- prompt and negative prompt;
- strength;
- guidance scale;
- inference steps;
- random seed;
- approval state and reviewer notes.

The default grid creates nine controlled candidates from three strengths and three guidance values.

## Curation criteria
Campaign approval checks:

1. product identity and silhouette;
2. material/color consistency;
3. no extra buttons, ports or accessories;
4. no duplicated product components;
5. no fake text, logos or watermarks;
6. product fully visible where required;
7. appropriate background and campaign composition;
8. no misleading functional implication.

## Promo video approach
The module takes two approved stills and writes 24 cross-fade PNG frames using `Image.blend`. A permitted video encoder can then encode the ordered frames to MP4/GIF. This method is intentionally simple and reproducible.

It should be described accurately as an assembled promo clip, not a learned video-generation model. True temporal video diffusion would require a dedicated pretrained video pipeline and substantially more GPU memory.

## Outputs

```text
generated/day16_variations/*.png
generated/day16_variations/asset_manifest.json
generated/day16_variations/promo_frames/frame_0000.png ...
```

## Run

```bash
python3 src/day16_image_variations_video.py
```

Generation requires a CUDA GPU and pretrained weights from Hugging Face or a local cache.

## Integration with Day 15
Day 15 produces original text-to-image marketing concepts. Day 16 can start from an approved real catalog image or an approved Day-15 concept, create controlled variants, curate the strongest assets and construct promo frames. Together these modules form RetailMind's first repeatable creative-production workflow.

## Day 17 handoff
The next generation-track increment can add text-to-speech responses and CLIP-based multimodal product search, connecting generated media with audio and cross-modal retrieval.
