# Day 11 - VAE Image Augmentation for Rare Product Categories

## Purpose
Day 11 strengthens RetailMind AI's Phase-2 catalog-vision system by generating additional images for an under-represented product category. The implementation uses a convolutional variational autoencoder, quality filtering, and a controlled before/after classifier evaluation.

## Why imbalance matters
A classifier optimized only for overall accuracy can perform poorly on rare classes while still reporting a strong aggregate score. For catalog operations, this creates uneven automation quality: common categories are tagged reliably, while uncommon categories require more manual correction. The relevant fairness target is therefore performance parity across product categories, especially minimum and rare-class accuracy.

## Dataset contract
Expected folder layout:

```text
data/fashion_products/<category>/*.jpg
```

The Kaggle Fashion Product Images Small dataset may require a one-time organization step from metadata into class folders. The pipeline accepts JPG, JPEG, and PNG images and uses the smallest non-empty folder as the default rare category.

## VAE architecture
- Input: 64×64 RGB image.
- Encoder: three strided convolution blocks with 32, 64, and 128 filters.
- Latent distribution: 64-dimensional mean and log-variance vectors.
- Sampling: reparameterization trick.
- Decoder: Dense reshape followed by three transposed-convolution blocks.
- Output: sigmoid RGB reconstruction.
- Objective: pixel reconstruction error plus beta-weighted KL divergence.

A VAE was chosen for stable CPU-oriented training. GANs can produce sharper images but are more vulnerable to unstable discriminator/generator balance and mode collapse.

## Quality and mode-collapse controls
Generated candidates are rejected when image standard deviation is below 0.06 or when maximum within-batch cosine similarity exceeds 0.985. These thresholds catch blank outputs and likely duplicates, but they do not guarantee realism. Human review should examine:

1. category correctness;
2. recognizable product silhouettes;
3. clipping and malformed edges;
4. texture and color plausibility;
5. repeated samples;
6. background artifacts.

Rejected outputs must never enter classifier training.

## Controlled augmentation experiment
The original and augmented classifiers must use the same architecture, validation split, seed, epoch budget, and optimization settings. The augmented dataset should be a copy of the original training tree with accepted images added only to the rare-class folder. Validation must contain real images only.

Report:
- overall validation accuracy;
- per-class accuracy;
- macro average per-class accuracy;
- rare-class accuracy;
- minimum per-class accuracy;
- accuracy lift and any regression by category.

This design separates genuine generalization improvement from memorization of synthetic artifacts.

## Generated artifacts
After successful execution and controlled augmentation:

- `models/rare_category_encoder_v11.keras`
- `models/rare_category_decoder_v11.keras`
- `models/catalog_classifier_augmented_v11.keras`
- `models/day11_per_class_accuracy.csv`
- `generated/day11_rare_category/*.png`

## Run

```bash
python3 src/day11_vae_image_augmentation.py
```

TensorFlow must be installed. The original dataset must exist at `data/fashion_products`. For the before/after comparison, prepare `data/fashion_products_augmented` with the same category names and only accepted synthetic additions.

## Fairness interpretation
A positive fairness result requires measurable improvement for the under-represented class and ideally an increase in macro or minimum-class accuracy. Overall accuracy alone is insufficient. Synthetic images can also encode unrealistic artifacts, amplify dataset bias, or shift the classifier toward generated textures, so all claims must be supported by untouched real validation data.

## Production recommendations
- Maintain an audit manifest linking each generated image to model version and random seed.
- Keep original and augmented datasets separate.
- Require catalog-quality review before acceptance.
- Use rolling classifier evaluations as new real product images arrive.
- Add conditional generation when multiple minority categories need separate augmentation.
- Consider FID/KID only when a sufficiently large real reference set is available; never replace downstream real-image accuracy with a generative-quality score.
