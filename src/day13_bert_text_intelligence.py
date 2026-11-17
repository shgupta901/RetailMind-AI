"""RetailMind AI - Day 13: BERT sentiment and support-ticket routing.

Primary library: Hugging Face Transformers (PyTorch backend).
Local CSV inputs:
- data/amazon_reviews.csv: text,label OR title,content,label
- data/support_tickets.csv: text,label OR title,content,label

The script fine-tunes two compact BERT-compatible classifiers, evaluates per-class
results, compares sentiment against the saved Day-12 metrics when available, and
persists deployable tokenizer/model folders.
"""

from pathlib import Path
import csv
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"
REVIEWS_PATH = "data/amazon_reviews.csv"
TICKETS_PATH = "data/support_tickets.csv"
SENTIMENT_DIR = "models/bert_sentiment_v13"
ROUTER_DIR = "models/bert_ticket_router_v13"
RESULTS_PATH = "models/day13_bert_metrics.json"
DAY12_METRICS_PATH = "models/day12_sentiment_metrics.csv"
MAX_LENGTH = 192
TRAIN_LIMIT = 12000
EVAL_LIMIT = 2500
EPOCHS = 2
BATCH_SIZE = 8
SEED = 42


def normalize_sentiment(value):
    value = str(value).strip().lower()
    if value in {"0", "negative", "neg", "false"}:
        return 0
    if value in {"1", "positive", "pos", "true"}:
        return 1
    if value == "2":
        return 1
    raise ValueError(f"Unsupported sentiment label: {value}")


def read_csv_records(path, task="sentiment", limit=None):
    records = []
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if "label" not in fields:
            raise ValueError(f"{path} must contain a label column")
        if "text" not in fields and "content" not in fields:
            raise ValueError(f"{path} must contain text or content")
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            text = f"{row.get('title', '')} {row.get('text', row.get('content', ''))}".strip()
            if not text:
                continue
            label = normalize_sentiment(row["label"]) if task == "sentiment" else str(row["label"]).strip()
            records.append((text, label))
    if len(records) < 20:
        raise ValueError(f"{path} needs at least 20 non-empty records")
    return records


def stratified_split(records, eval_ratio=0.2):
    grouped = {}
    for text, label in records:
        grouped.setdefault(label, []).append((text, label))
    rng = random.Random(SEED)
    train, evaluation = [], []
    for group in grouped.values():
        rng.shuffle(group)
        cut = max(1, int(len(group) * eval_ratio))
        evaluation.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


class EncodedTextDataset(Dataset):
    def __init__(self, records, tokenizer, label_to_id):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        text, label = self.records[index]
        encoded = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, padding=False
        )
        encoded["labels"] = self.label_to_id[label]
        return encoded


def classification_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=-1)
    label_count = int(max(labels.max(), predictions.max()) + 1)
    confusion = [[0 for _ in range(label_count)] for _ in range(label_count)]
    for truth, pred in zip(labels.tolist(), predictions.tolist()):
        confusion[truth][pred] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(label_count))
    metrics = {"accuracy": correct / total if total else 0.0}
    f1_values = []
    for label in range(label_count):
        tp = confusion[label][label]
        fp = sum(confusion[row][label] for row in range(label_count) if row != label)
        fn = sum(confusion[label][col] for col in range(label_count) if col != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"class_{label}_precision"] = precision
        metrics[f"class_{label}_recall"] = recall
        metrics[f"class_{label}_f1"] = f1
        f1_values.append(f1)
    metrics["macro_f1"] = sum(f1_values) / len(f1_values)
    return metrics


def train_classifier(records, output_dir, task_name):
    labels = sorted({label for _, label in records}, key=str)
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: str(label) for label, i in label_to_id.items()}
    train_records, eval_records = stratified_split(records)
    train_records = train_records[:TRAIN_LIMIT]
    eval_records = eval_records[:EVAL_LIMIT]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id={str(k): v for k, v in label_to_id.items()},
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{output_dir}_checkpoints",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            seed=SEED,
        ),
        train_dataset=EncodedTextDataset(train_records, tokenizer, label_to_id),
        eval_dataset=EncodedTextDataset(eval_records, tokenizer, label_to_id),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=classification_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "label_mapping.json", "w", encoding="utf-8") as file:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, file, indent=2)
    print(task_name, metrics)
    return metrics


def load_day12_metrics(path=DAY12_METRICS_PATH):
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(file)}


def route_ticket(text, model_dir=ROUTER_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    index = int(probabilities.argmax())
    return {"route": model.config.id2label[index], "confidence": float(probabilities[index])}


def main():
    set_seed(SEED)
    reviews = read_csv_records(REVIEWS_PATH, task="sentiment")
    tickets = read_csv_records(TICKETS_PATH, task="routing")
    print("Review labels:", Counter(label for _, label in reviews))
    print("Ticket labels:", Counter(label for _, label in tickets))

    sentiment_metrics = train_classifier(reviews, SENTIMENT_DIR, "sentiment")
    routing_metrics = train_classifier(tickets, ROUTER_DIR, "ticket routing")
    day12 = load_day12_metrics()

    comparison = {
        "day12_classical": day12,
        "day13_bert_sentiment": sentiment_metrics,
        "day13_bert_ticket_routing": routing_metrics,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
    }
    if day12.get("f1") is not None and sentiment_metrics.get("eval_macro_f1") is not None:
        comparison["bert_f1_lift_vs_day12"] = sentiment_metrics["eval_macro_f1"] - day12["f1"]

    Path("models").mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
    print("Saved:", SENTIMENT_DIR, ROUTER_DIR, RESULTS_PATH)


if __name__ == "__main__":
    main()
