"""RetailMind AI - Day 14: review summaries and grounded support chatbot.

Libraries/services:
- Hugging Face Transformers for local review summarization.
- OpenAI or Anthropic HTTP API for grounded support answers.
- Python standard library for CSV, JSON, environment variables, and HTTP.

Secrets are read only from environment variables. No API key is hard-coded.
"""

from pathlib import Path
import csv
import json
import os
import textwrap
import urllib.error
import urllib.request
from transformers import pipeline

REVIEWS_PATH = "data/product_reviews.csv"
KNOWLEDGE_PATH = "data/product_policy_knowledge.json"
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
SUMMARY_OUTPUT = "models/day14_product_review_summaries.json"
CHAT_LOG_PATH = "models/day14_chat_transcript.json"
MAX_REVIEWS_PER_PRODUCT = 40
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are RetailMind Assistant, a concise and professional retail support assistant.
Answer only from the supplied product and policy context. Do not invent prices, delivery dates,
stock, warranties, refunds, or product capabilities. If the context does not contain the answer,
say that the information is unavailable and recommend human support. Do not request passwords,
full payment-card details, or other sensitive credentials. Keep the response under 140 words."""


def load_reviews(path=REVIEWS_PATH):
    grouped = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if "product_id" not in fields:
            raise ValueError("Reviews CSV must contain product_id")
        if "review" not in fields and "text" not in fields:
            raise ValueError("Reviews CSV must contain review or text")
        for row in reader:
            text = row.get("review", row.get("text", "")).strip()
            if text:
                grouped.setdefault(row["product_id"], []).append(text)
    if not grouped:
        raise ValueError("No valid reviews found")
    return grouped


def chunk_reviews(reviews, max_chars=3200):
    chunks, current, size = [], [], 0
    for review in reviews:
        cleaned = " ".join(review.split())
        if current and size + len(cleaned) > max_chars:
            chunks.append(" Reviews: ".join(current))
            current, size = [], 0
        current.append(cleaned)
        size += len(cleaned)
    if current:
        chunks.append(" Reviews: ".join(current))
    return chunks


def build_summarizer(model_name=SUMMARY_MODEL):
    return pipeline("summarization", model=model_name, tokenizer=model_name)


def summarize_product_reviews(grouped_reviews, summarizer):
    results = {}
    for product_id, reviews in grouped_reviews.items():
        selected = reviews[:MAX_REVIEWS_PER_PRODUCT]
        partials = []
        for chunk in chunk_reviews(selected):
            prompt = (
                "Summarize the recurring product strengths, complaints, and customer requests "
                "from these reviews. Do not add facts. " + chunk
            )
            output = summarizer(
                prompt, max_length=130, min_length=35, do_sample=False,
                truncation=True
            )[0]["summary_text"]
            partials.append(output)
        combined = " ".join(partials)
        if len(partials) > 1:
            combined = summarizer(
                "Consolidate these review summaries without inventing facts: " + combined,
                max_length=150, min_length=45, do_sample=False, truncation=True
            )[0]["summary_text"]
        results[product_id] = {
            "review_count": len(reviews),
            "reviews_summarized": len(selected),
            "summary": combined,
        }
    return results


def load_knowledge(path=KNOWLEDGE_PATH):
    with open(path, encoding="utf-8") as file:
        knowledge = json.load(file)
    if not isinstance(knowledge, dict):
        raise ValueError("Knowledge JSON must be an object")
    return knowledge


def select_context(question, knowledge, max_sections=4):
    """Simple lexical retrieval so only relevant product/policy text reaches the LLM."""
    query_terms = {word.strip(".,?!:;()[]\"").lower() for word in question.split() if len(word) > 2}
    candidates = []
    sections = knowledge.get("sections", knowledge)
    iterable = sections.items() if isinstance(sections, dict) else [
        (item.get("title", f"section_{i}"), item.get("text", ""))
        for i, item in enumerate(sections)
    ]
    for title, text in iterable:
        body = text if isinstance(text, str) else json.dumps(text)
        haystack = f"{title} {body}".lower()
        score = sum(term in haystack for term in query_terms)
        candidates.append((score, str(title), body))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [item for item in candidates if item[0] > 0][:max_sections]
    if not selected:
        selected = candidates[:2]
    return "\n\n".join(f"[{title}]\n{text}" for _, title, text in selected)


def openai_chat(messages, model=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = {
        "model": model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 250,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"].strip()


def anthropic_chat(messages, model=None):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    user_messages = [m for m in messages if m["role"] != "system"]
    payload = {
        "model": model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        "system": system,
        "messages": user_messages,
        "temperature": 0.1,
        "max_tokens": 250,
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    return data["content"][0]["text"].strip()


def call_llm(messages, provider=None):
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    try:
        if provider == "openai":
            return openai_chat(messages)
        if provider == "anthropic":
            return anthropic_chat(messages)
        raise ValueError("LLM_PROVIDER must be openai or anthropic")
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API request failed ({error.code}): {details}") from error


class RetailMindChatbot:
    def __init__(self, knowledge, provider=None):
        self.knowledge = knowledge
        self.provider = provider
        self.history = []

    def ask(self, question):
        context = select_context(question, self.knowledge)
        grounded_question = f"""PRODUCT/POLICY CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

Answer only from the context. If evidence is missing, say so clearly."""
        recent = self.history[-MAX_HISTORY_TURNS * 2:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent + [
            {"role": "user", "content": grounded_question}
        ]
        answer = call_llm(messages, self.provider)
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return {"answer": answer, "context": context}

    def save_history(self, path=CHAT_LOG_PATH):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)


def main():
    reviews = load_reviews()
    summarizer = build_summarizer()
    summaries = summarize_product_reviews(reviews, summarizer)
    Path("models").mkdir(exist_ok=True)
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    print("Saved review summaries:", SUMMARY_OUTPUT)

    knowledge = load_knowledge()
    bot = RetailMindChatbot(knowledge)
    print("RetailMind Assistant ready. Type 'quit' to exit.")
    while True:
        question = input("Customer: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        result = bot.ask(question)
        print("Assistant:", textwrap.fill(result["answer"], width=90))
    bot.save_history()
    print("Saved conversation:", CHAT_LOG_PATH)


if __name__ == "__main__":
    main()
