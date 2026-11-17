# Day 13 - BERT Text Intelligence

## Purpose
Day 13 upgrades RetailMind AI's classical review sentiment baseline with a pretrained Transformer and adds multi-class support-ticket routing. Two separately fine-tuned `distilbert-base-uncased` classifiers are persisted for deployment.

## Why DistilBERT
DistilBERT retains contextual bidirectional language understanding while being smaller and faster than BERT-base. It is suitable for a hosted learning environment, although CPU fine-tuning can still be slow. The default configuration uses sequence length 192, batch size 8, learning rate `2e-5`, and two epochs.

## Inputs
Place these local files in the repository:

```text
data/amazon_reviews.csv
data/support_tickets.csv
```

Both accept `text,label` or `title,content,label`. Amazon sentiment labels may be `0/1`, `negative/positive`, or `1/2`. Ticket labels should be meaningful route names such as `delivery`, `refund`, `damaged_item`, and `account_support`.

## Tokenization
The Hugging Face tokenizer:

1. adds model-specific special tokens;
2. applies WordPiece-compatible subword tokenization;
3. returns input IDs and attention masks;
4. truncates sequences at 192 tokens;
5. dynamically pads each batch through `DataCollatorWithPadding`.

Dynamic padding reduces unnecessary computation compared with padding every record to the maximum length.

## Fine-tuning design
Each task receives its own `AutoModelForSequenceClassification` head. Stratified splitting preserves class representation. Training selects the best epoch by macro-F1 and uses weight decay of 0.01. Training and evaluation limits keep the project practical, but full experiments should document any sampling.

## Sentiment comparison with Day 12
The script reads `models/day12_sentiment_metrics.csv` when available and records the Day-13 macro-F1 lift. A valid comparison should use the same held-out examples, label mapping, and exclusion rules. BERT should replace the active baseline only when quality improvement justifies:

- larger disk and memory footprint;
- slower training and inference;
- pretrained-model download/caching requirements;
- more complex monitoring and deployment.

## Ticket-routing evaluation
Report overall accuracy, macro-F1, and per-class precision/recall/F1. Macro-F1 prevents frequent ticket types from dominating the result. Class-level recall is especially important because low recall means valid tickets for that team are missed.

## Confidence policy
The inference helper returns route and confidence. A production system should calibrate a threshold on validation data. For example, automatic routing may be enabled above a validated 0.80 threshold, with lower-confidence tickets sent to manual triage. The value 0.80 is an operational starting point, not a universal guarantee.

## Generated artifacts

```text
models/bert_sentiment_v13/
models/bert_ticket_router_v13/
models/day13_bert_metrics.json
```

Each model folder contains model weights, tokenizer files, configuration, and a label-mapping JSON file.

## Run

```bash
python3 src/day13_bert_text_intelligence.py
```

The first execution requires the pretrained model to be available from Hugging Face or already present in the local cache. The environment also requires the PyTorch backend used by Transformers.

## Deployment recommendation
Keep Day 12 as a fallback. Log model version, route/sentiment, confidence, latency, and human correction. Monitor class distribution, macro-F1, low-confidence rate, and new ticket categories. Retrain only through a versioned evaluation process.

## Risks and limitations
- pretrained and fine-tuning data can introduce bias;
- long reviews may lose decisive text through truncation;
- confidence may be poorly calibrated;
- category drift can invalidate the routing head;
- ticket labels fabricated from reviews may not represent real support operations;
- model outputs should assist rather than silently override high-impact support workflows.

## Day 14 handoff
Day 14 can use an LLM API for review summarization and chatbot responses while retaining BERT sentiment/routing as deterministic classification guardrails and Day 12 as the low-cost benchmark.
