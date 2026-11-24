"""RetailMind AI - Day 17: TTS voice replies and CLIP multimodal product search.

Libraries: Hugging Face Transformers (TTS + CLIP), PyTorch, standard library.
Dataset: Fashion Product Images Small plus local catalog text metadata.

Expected files:
- data/day17_catalog.json
- image files referenced by image_path in the JSON
"""

from pathlib import Path
import json
import wave
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
TTS_MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
CATALOG_PATH = "data/day17_catalog.json"
INDEX_PATH = "models/day17_clip_catalog_index.npz"
METADATA_INDEX_PATH = "models/day17_clip_catalog_metadata.json"
VOICE_OUTPUT_DIR = "generated/day17_voice"
TOP_K = 5
SEED = 42


def load_catalog(path=CATALOG_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    products = data.get("products", data if isinstance(data, list) else [])
    if not products:
        raise ValueError("Catalog JSON must contain a non-empty products list.")
    required = {"product_id", "product_name", "description", "image_path"}
    for product in products:
        missing = required.difference(product)
        if missing:
            raise ValueError(f"Missing product fields: {sorted(missing)}")
        if not Path(product["image_path"]).exists():
            raise FileNotFoundError(product["image_path"])
    return products


def load_clip(model_id=CLIP_MODEL_ID, local_files_only=False):
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    return model, processor


def normalize(vectors):
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_catalog_index(products, model, processor):
    images = [Image.open(p["image_path"]).convert("RGB") for p in products]
    texts = [f"{p['product_name']}. {p['description']}" for p in products]
    with torch.no_grad():
        image_inputs = processor(images=images, return_tensors="pt", padding=True)
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        image_embeddings = normalize(model.get_image_features(**image_inputs))
        text_embeddings = normalize(model.get_text_features(**text_inputs))
    # Combined catalog vector supports a product representation grounded in both modalities.
    combined_embeddings = normalize((image_embeddings + text_embeddings) / 2.0)
    return image_embeddings, text_embeddings, combined_embeddings


def save_index(products, image_embeddings, text_embeddings, combined_embeddings):
    Path("models").mkdir(exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        image=image_embeddings.cpu().numpy(),
        text=text_embeddings.cpu().numpy(),
        combined=combined_embeddings.cpu().numpy(),
    )
    Path(METADATA_INDEX_PATH).write_text(json.dumps(products, indent=2), encoding="utf-8")


def load_saved_index():
    arrays = np.load(INDEX_PATH)
    products = json.loads(Path(METADATA_INDEX_PATH).read_text(encoding="utf-8"))
    return products, {name: arrays[name] for name in arrays.files}


def rank_products(query_vector, catalog_vectors, products, k=TOP_K):
    scores = np.asarray(catalog_vectors) @ np.asarray(query_vector).reshape(-1)
    order = np.argsort(scores)[::-1][:k]
    return [
        {
            "rank": rank + 1,
            "score": float(scores[index]),
            "product_id": products[index]["product_id"],
            "product_name": products[index]["product_name"],
            "description": products[index]["description"],
            "image_path": products[index]["image_path"],
        }
        for rank, index in enumerate(order)
    ]


def search_by_text(query, model, processor, products, catalog_vectors, k=TOP_K):
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        query_vector = normalize(model.get_text_features(**inputs))[0].cpu().numpy()
    return rank_products(query_vector, catalog_vectors, products, k)


def search_by_image(image_path, model, processor, products, catalog_vectors, k=TOP_K):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt")
        query_vector = normalize(model.get_image_features(**inputs))[0].cpu().numpy()
    return rank_products(query_vector, catalog_vectors, products, k)


def load_tts(local_files_only=False):
    synthesizer = pipeline(
        "text-to-speech",
        model=TTS_MODEL_ID,
        vocoder=VOCODER_ID,
        device=-1,
        model_kwargs={"local_files_only": local_files_only},
    )
    return synthesizer


def save_waveform(audio, sampling_rate, path):
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sampling_rate))
        wav.writeframes(pcm.tobytes())
    return path


def speak_reply(text, synthesizer, filename="support_reply.wav"):
    clean_text = " ".join(str(text).split())
    if len(clean_text) > 500:
        clean_text = clean_text[:500].rsplit(" ", 1)[0] + "."
    result = synthesizer(clean_text)
    output = Path(VOICE_OUTPUT_DIR) / filename
    save_waveform(result["audio"], result["sampling_rate"], output)
    return output


def assistant_multimodal_response(question, support_answer, model, processor,
                                  products, catalog_vectors, synthesizer=None):
    """Bridge Day 14 support text to Day 17 voice and optional relevant-product search."""
    matches = search_by_text(question, model, processor, products, catalog_vectors, k=3)
    voice_path = None
    if synthesizer is not None:
        voice_path = str(speak_reply(support_answer, synthesizer, "assistant_reply.wav"))
    return {"support_answer": support_answer, "voice_path": voice_path, "related_products": matches}


def main():
    torch.manual_seed(SEED)
    products = load_catalog()
    clip_model, clip_processor = load_clip()
    image_emb, text_emb, combined_emb = build_catalog_index(products, clip_model, clip_processor)
    save_index(products, image_emb, text_emb, combined_emb)
    print("Indexed products:", len(products))
    print("Saved:", INDEX_PATH)
    print("Saved:", METADATA_INDEX_PATH)
    print("Example text search:")
    for result in search_by_text(
        "lightweight black running shoes", clip_model, clip_processor,
        products, combined_emb.cpu().numpy(), k=min(3, len(products))
    ):
        print(result["rank"], result["product_name"], round(result["score"], 4))


if __name__ == "__main__":
    main()
