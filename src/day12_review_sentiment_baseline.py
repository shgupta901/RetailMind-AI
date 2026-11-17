"""RetailMind AI - Day 12: classical review sentiment analytics.

Libraries used for ML/NLP: scikit-learn and NLTK.
Expected CSV columns:
- text, label
or
- title, content, label
Labels may be 0/1, negative/positive, or 1/2.
"""

from pathlib import Path
import csv
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/amazon_reviews.csv"
MODEL_PATH = "models/review_sentiment_tfidf_v12.pkl"
TERMS_PATH = "models/day12_informative_terms.csv"
METRICS_PATH = "models/day12_sentiment_metrics.csv"
RANDOM_STATE = 42
MAX_FEATURES = 30000
TOP_TERMS = 25

# Compact fallback list keeps the pipeline usable when hosted Jupyter blocks NLTK downloads.
FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
    "she", "so", "that", "the", "their", "theirs", "them", "they", "this",
    "to", "too", "was", "we", "were", "what", "when", "where", "which",
    "who", "will", "with", "you", "your", "yours"
}


def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


def normalize_label(value):
    text = str(value).strip().lower()
    if text in {"1", "positive", "pos", "true"}:
        return 1
    if text in {"0", "negative", "neg", "false"}:
        return 0
    if text == "2":  # amazon_polarity often uses class 1/2 in exported files
        return 1
    raise ValueError(f"Unsupported sentiment label: {value}")


def load_reviews(path=DATA_PATH, limit=None):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if "label" not in fields:
            raise ValueError("CSV must contain a label column.")
        if "text" not in fields and "content" not in fields:
            raise ValueError("CSV must contain text or content.")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            title = row.get("title", "")
            body = row.get("text", row.get("content", ""))
            combined = f"{title} {body}".strip()
            if combined:
                texts.append(combined)
                labels.append(normalize_label(row["label"]))
    if len(texts) < 20 or len(set(labels)) < 2:
        raise ValueError("Need at least 20 reviews containing both sentiment classes.")
    return texts, labels


class ReviewCleaner(BaseEstimator, TransformerMixin):
    """Pickle-safe scikit-learn transformer using NLTK tokenization/lemmatization."""
    def __init__(self, lemmatize=True):
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        self.stop_words_ = get_stopwords()
        self.lemmatizer_ = WordNetLemmatizer()
        return self

    def transform(self, X):
        cleaned = []
        for review in X:
            text = str(review).lower()
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"[^a-z\s']", " ", text)
            tokens = [t for t in wordpunct_tokenize(text) if t.isalpha() and len(t) > 1]
            tokens = [t for t in tokens if t not in self.stop_words_]
            if self.lemmatize:
                output = []
                for token in tokens:
                    try:
                        output.append(self.lemmatizer_.lemmatize(token))
                    except LookupError:
                        output.append(token)
                tokens = output
            cleaned.append(" ".join(tokens))
        return cleaned


def build_pipeline(model="logistic"):
    if model == "logistic":
        classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )
    else:
        raise ValueError("Supported model: logistic")
    return Pipeline([
        ("cleaner", ReviewCleaner()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95,
            max_features=MAX_FEATURES, sublinear_tf=True
        )),
        ("classifier", classifier),
    ])


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
    }
    return metrics, confusion_matrix(y_test, pred), classification_report(
        y_test, pred, target_names=["negative", "positive"], zero_division=0
    )


def informative_terms(model, top_n=TOP_TERMS):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    positive_indices = coefficients.argsort()[-top_n:][::-1]
    negative_indices = coefficients.argsort()[:top_n]
    positive = [(features[i], float(coefficients[i])) for i in positive_indices]
    negative = [(features[i], float(coefficients[i])) for i in negative_indices]
    return positive, negative


def save_terms(positive, negative, path=TERMS_PATH):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sentiment", "term", "coefficient"])
        for term, score in positive:
            writer.writerow(["positive", term, score])
        for term, score in negative:
            writer.writerow(["negative", term, score])


def predict_review(model, review):
    prediction = int(model.predict([review])[0])
    probability = float(model.predict_proba([review])[0, 1])
    return {
        "sentiment": "positive" if prediction == 1 else "negative",
        "positive_probability": probability,
    }


def main():
    texts, labels = load_reviews()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    metrics, matrix, report = evaluate(pipeline, x_test, y_test)

    print("Reviews:", len(texts), "Train:", len(x_train), "Test:", len(x_test))
    print("Confusion matrix:\n", matrix)
    print(report)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    positive, negative = informative_terms(pipeline)
    print("Top positive terms:", positive[:10])
    print("Top negative terms:", negative[:10])

    Path("models").mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(pipeline, file)
    save_terms(positive, negative)
    with open(METRICS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    print("Saved:", MODEL_PATH)
    print("Saved:", TERMS_PATH)
    print("Saved:", METRICS_PATH)


if __name__ == "__main__":
    main()
