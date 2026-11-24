# Day 17 - TTS Voice and CLIP Multimodal Product Search

## Purpose
Day 17 completes RetailMind AI's generation track by adding voice output to the support assistant and CLIP-based product retrieval across text and images.

## Dataset
The product search uses Fashion Product Images Small or approved catalog imagery plus local text metadata. Each product needs:

- `product_id`;
- `product_name`;
- `description`;
- `image_path`.

The image and description refer to the same catalog item and share one stable product ID.

## CLIP model
The default is `openai/clip-vit-base-patch32`. CLIP provides a text encoder and vision encoder trained so the two modalities occupy a compatible semantic space.

RetailMind stores:

1. normalized image embeddings;
2. normalized text embeddings;
3. a normalized average used as the default combined product vector.

This supports text-to-product and image-to-product retrieval with the same index.

## Text search
A natural-language product description is encoded with the CLIP text encoder. The resulting vector is compared with normalized catalog vectors using dot-product cosine similarity. Top-K products are returned with scores and catalog metadata.

## Image search
A reference image is encoded using CLIP's vision branch. The resulting vector is ranked against the same catalog index. This supports "find products that look like this" interactions.

## Retrieval evaluation
Search quality should use labeled queries rather than anecdotal examples. For a set of known relevant product IDs, report:

- Hit@1;
- Recall@5;
- Recall@10;
- text-query and image-query results separately.

A relevant result at rank 20 is operationally different from rank 1 even if both are technically retrieved.

## TTS model
The module uses Hugging Face `microsoft/speecht5_tts` with `microsoft/speecht5_hifigan` through the Transformers text-to-speech pipeline. Day-14 grounded support text becomes the TTS input, so voice does not create a second independent policy answer.

Generated audio is saved as 16-bit mono WAV under:

```text
generated/day17_voice/
```

## Accessibility and voice review
TTS should supplement the visible transcript. Validate intelligibility, brand/product pronunciation, number/date/currency reading, clipping, response length and replay controls. Do not hide the written policy answer solely because voice is available.

## Index artifacts

```text
models/day17_clip_catalog_index.npz
models/day17_clip_catalog_metadata.json
```

The index should be regenerated whenever descriptions or product images change materially.

## Run

```bash
python3 src/day17_tts_multimodal_search.py
```

The first model use may require Hugging Face access unless the CLIP/TTS models are already cached. CLIP indexing can run on CPU for a small learning catalog; production-scale indexing and low-latency search require capacity planning.

## Integration
The Day-14 chatbot returns grounded text. Day 17 can convert that response to audio and separately use the customer's question or image to retrieve related products. Exact business filters such as price, inventory, size and region should remain structured filters rather than relying only on semantic similarity.

## Risks and limitations
- CLIP can prefer visual similarity over functional equivalence.
- Product descriptions may bias the combined vector toward marketing language.
- Similarity scores are relative ranking signals, not calibrated probabilities.
- TTS may mispronounce SKU codes or specialized product names.
- Catalog and model updates require index versioning.
- Licensing, security and accessibility review remain required before deployment.

## Phase transition
Day 17 completes the text/image/video/voice/multimodal generation track. The next phase can reuse embedding and retrieval concepts to build a vector-database knowledge base for grounded enterprise retrieval.
