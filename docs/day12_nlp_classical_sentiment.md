# Day 12 - Classical NLP Review Sentiment Baseline

## Purpose
Day 12 creates RetailMind AI's first review-analytics component. It transforms Amazon product-review text into sentiment predictions and interpretable terms using NLTK cleaning, TF-IDF features, and scikit-learn logistic regression.

## Business value
Customer reviews contain direct evidence about product quality, delivery, packaging, support, fit, and value. A sentiment baseline supports:

- negative-review monitoring;
- product issue discovery;
- category-level mood tracking;
- prioritization of manual review analysis;
- a measurable benchmark for the later BERT upgrade.

The output is a signal for investigation, not an automatic statement of customer intent.

## Dataset contract
Place a local export at:

```text
data/amazon_reviews.csv
```

Accepted schemas:

```text
text,label
```

or

```text
title,content,label
```

Supported labels are `0/1`, `negative/positive`, and `1/2`. The loader requires both classes and at least 20 non-empty reviews. For meaningful evaluation, several thousand balanced reviews are recommended.

## Cleaning pipeline
The pickle-safe `ReviewCleaner` transformer performs lowercasing, URL removal, HTML removal, alphabetic filtering, NLTK `wordpunct_tokenize`, stopword removal, and optional WordNet lemmatization. A fallback stopword set prevents total failure in offline environments without downloaded NLTK resources.

Important trade-offs:

- Removing punctuation simplifies noise but can remove emphasis.
- Stopword removal may delete useful negation words if configured carelessly.
- Lemmatization reduces vocabulary size but can blur domain-specific forms.
- English-only cleaning is unsuitable for multilingual reviews.

## TF-IDF design
The vectorizer uses:

- unigram and bigram features;
- maximum 30,000 features;
- minimum document frequency of 2;
- maximum document frequency of 0.95;
- sublinear term frequency.

Bigrams preserve short phrases such as `poor quality` and `highly recommend`. TF-IDF remains sparse, fast, explainable, and appropriate for a classical baseline.

## Classifier
Balanced logistic regression is used because it handles high-dimensional sparse matrices, produces probabilities, and exposes signed feature coefficients. The full cleaner-vectorizer-classifier chain is stored as one scikit-learn pipeline, preventing training/inference preprocessing drift.

## Evaluation
The split is stratified 80/20 with random state 42. Report:

- accuracy;
- positive-class precision;
- positive-class recall;
- F1;
- ROC-AUC;
- confusion matrix;
- classification report.

Inspect false positives and false negatives. Accuracy alone is insufficient when class balance or business costs differ. If negative reviews are escalation triggers, missing a negative review may matter more than incorrectly flagging a positive review.

## Informative terms
Logistic-regression coefficients are exported to:

```text
models/day12_informative_terms.csv
```

Large positive coefficients are associated with positive predictions and large negative coefficients with negative predictions. These are correlations learned from the sample, not causal explanations. Terms may reflect brands, product mix, or annotation artifacts.

## Generated artifacts
After successful execution:

- `models/review_sentiment_tfidf_v12.pkl`
- `models/day12_informative_terms.csv`
- `models/day12_sentiment_metrics.csv`

## Run

```bash
python3 src/day12_review_sentiment_baseline.py
```

Install scikit-learn and NLTK in the execution environment and place the CSV at the required path. Never load an untrusted pickle.

## Production considerations
- Monitor label and vocabulary drift by product category.
- Keep the original held-out split for the Day 13 BERT comparison.
- Add threshold calibration when false-negative and false-positive costs differ.
- Audit influential terms for brand leakage and protected-attribute proxies.
- Retain review IDs outside the model artifact for traceability.
- Add language detection before using an English-only cleaner.
- Compare macro-F1 and category-level metrics, not only overall accuracy.

## Day 13 handoff
Day 13 can upgrade contextual understanding with BERT while preserving this classical pipeline as a speed, cost, and interpretability benchmark. BERT must demonstrate measurable improvement on the same untouched evaluation data.
