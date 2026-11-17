# Day 14 - LLM Review Summaries and Support Chatbot

## Purpose
Day 14 creates RetailMind AI's first assistant module. A Hugging Face summarizer condenses product-review themes, and a provider-neutral LLM client answers customers using selected product and policy context.

## Components

### Review summarizer
- Groups reviews by `product_id`.
- Limits processing to 40 reviews per product by default.
- Chunks long review groups.
- Uses `sshleifer/distilbart-cnn-12-6` through the Transformers summarization pipeline.
- Produces product-level JSON with review counts and summaries.

### Grounded support chatbot
- Reads product/policy sections from local JSON.
- Selects context through lexical overlap with the customer question.
- Supports OpenAI and Anthropic HTTP APIs.
- Maintains a bounded six-turn conversation history.
- Requires explicit refusal/escalation when context is insufficient.
- Returns both answer and grounding context for audit.

## Input contracts

`data/product_reviews.csv`:

```text
product_id,review
P1001,"Comfortable and well made..."
```

`data/product_policy_knowledge.json` example:

```json
{
  "sections": {
    "returns": "Unused products may be returned within the stated policy period...",
    "damaged_items": "Photograph damaged packaging and contact support...",
    "product_P1001": "Product description and verified specifications..."
  }
}
```

Only approved product/policy text should enter this file.

## Secret management
Never hard-code or commit API keys. Configure exactly one provider through environment variables:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
```

or:

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="claude-3-5-haiku-latest"
```

Do not store `.env` files in Git unless they contain placeholders only and are protected by `.gitignore`.

## Grounding design
The lexical selector scores each section by overlap with meaningful question terms and sends only the top sections. This reduces irrelevant context and API token usage. It is intentionally simple. A future RAG increment should add embeddings, a vector database, metadata filters, citations, and retrieval evaluation.

## Prompt controls
The system prompt requires the assistant to:

- answer from supplied context only;
- avoid inventing price, stock, delivery, refund, warranty, or capability claims;
- disclose when information is unavailable;
- recommend human support when evidence is missing;
- avoid requesting passwords or complete payment-card details;
- remain professional and concise.

Prompt instructions reduce risk but are not a security boundary. Validate outputs and restrict downstream actions.

## Evaluation checklist
Review-summary evaluation:

- factual consistency with source reviews;
- coverage of major positive and negative themes;
- preservation of safety/defect complaints;
- absence of invented counts and attributes;
- no unnecessary reviewer personal information.

Chatbot evaluation:

- grounded-answer rate;
- unsupported-claim rate;
- correct escalation rate;
- prompt-injection resistance;
- response latency;
- token cost;
- usefulness judged by support reviewers.

Use a fixed test set of answerable, unanswerable, adversarial, and multi-turn questions.

## Generated artifacts

- `models/day14_product_review_summaries.json`
- `models/day14_chat_transcript.json`

Review transcripts before long-term retention. Remove personal data that is not needed for support auditing.

## Run

```bash
python3 src/day14_llm_summaries_chatbot.py
```

The summarizer may need to download its Hugging Face model on first use. The chatbot requires network access and a valid provider API key.

## Operational safeguards
- Set API budgets and rate limits.
- Use a manual-support fallback.
- Log model/provider version and selected knowledge sections.
- Avoid executing refunds, order changes, or account actions from free-form model text.
- Apply authentication and authorization before exposing customer/order data.
- Redact unnecessary personal information before sending prompts externally.
- Re-test whenever prompts, models, policies, or providers change.

## Day 15 handoff
Day 15 adds Stable Diffusion for on-brand marketing imagery. The assistant module should remain separate from image generation so text support policies and creative-generation controls can be governed independently.
