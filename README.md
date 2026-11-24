# RetailMind AI - Enterprise Retail Intelligence Platform

RetailMind AI is an end-to-end retail intelligence platform developed incrementally across 35 production milestones. It combines predictive churn analytics, deep learning, computer vision, explainable AI, and demand forecasting in one evolving codebase.

## Phase 1 & Phase 2 Master Milestone Ledger

### Part A: Phase 1 - Classical ML & Deep Churn Engine (Days 01-05)

| Day | Focus / Architecture | Accuracy | Recall (Churn=1) | Precision (Churn=1) | ROC-AUC | Status |
|---|---|---:|---:|---:|---:|---|
| **01** | Baseline Logistic Regression Pipeline | 0.8053 | 0.5722 | 0.6544 | 0.8361 | Deprecated |
| **02** | Decision Tree / Random Forest / SVM | 0.7854 | 0.7567 | 0.5739 | 0.8491 | Evaluated |
| **03** | Tuned Random Forest (5-Fold CV) | 0.7925 | 0.7701 | 0.5842 | 0.8498 | Audited |
| **04** | Feature Engineering + XGBoost Ensemble | **0.8010** | **0.8048** | **0.5960** | **0.8612** | **Production Champion** |
| **05** | Deep ANN (Keras + BatchNorm + Dropout) | 0.7583 | 0.7834 | 0.5288 | 0.8415 | DL Benchmark |

### Part B: Phase 2 - Computer Vision & Catalog Intelligence (Days 06-08)

| Day | Vision Module | Target Architecture | Accuracy / Score | Primary Business Value | Status |
|---|---|---|---:|---|---|
| **06** | Product Auto-Tagger | Sequential CNN (Conv2D + MaxPool) | ~0.90 | Automates manual catalog data entry | **V1 Shipped** |
| **07** | Transfer Learning & Detection | MobileNetV2 Backbone + YOLOv8 | ~0.94 | High-resolution RGB classification & multi-item QA | **Shipped** |
| **08** | Packaged CV Module | OpenCV GrabCut + CNN + Grad-CAM | ~0.90 | Background removal & explainable AI | **Phase 2 Complete** |

### Part C: Phase 3 - Sequence Forecasting (Days 09-10)

| Day | Forecasting Module | Architecture | Horizon | Evaluation | Status |
|---|---|---|---|---|---|
| **09** | Single-Step Demand Forecasting | LSTM (64 units) | Next day | MAE / RMSE | **Integrated** |
| **10** | Multi-Horizon Demand Forecasting | Seq2Seq LSTM + Attention | Next 14 days | MAE / RMSE / MAPE | **Integrated** |

### Part D: Phase 4 - Generative Catalog Augmentation (Day 11)

| Day | Module | Architecture | Business Goal | Evaluation | Status |
|---|---|---|---|---|---|
| **11** | Rare-Category Image Augmentation | Convolutional VAE | Improve class balance and catalog fairness | Per-class accuracy lift + quality review | **Integrated** |

### Part E: Phase 5 - Text Intelligence & Assistant Apps (Days 12-14)

| Day | Module | Architecture | Business Goal | Evaluation | Status |
|---|---|---|---|---|---|
| **12** | Review Sentiment Baseline | NLTK + TF-IDF + Logistic Regression | Convert reviews into customer-mood and product-issue signals | Accuracy / Precision / Recall / F1 / ROC-AUC | **Integrated** |
| **13** | Contextual Sentiment & Ticket Routing | DistilBERT sequence classifiers | Improve sentiment accuracy and automate support routing | Accuracy / Macro-F1 / Per-class Precision-Recall | **Integrated** |
| **14** | Review Summaries & Support Assistant | Hugging Face summarizer + grounded LLM API client | Reduce review-reading and support-response workload | Faithfulness / Grounded-answer rate / Escalation / Latency | **Integrated** |

### Part F: Phase 6 - Generative Marketing (Days 15-16)

| Day | Module | Architecture | Business Goal | Controls | Status |
|---|---|---|---|---|---|
| **15** | On-Brand Marketing Imagery | Stable Diffusion v1.5 via diffusers | Generate reusable campaign concepts from product metadata | Prompt templates / Negative prompts / Fixed seeds / Human approval | **Integrated** |
| **16** | Product Variations & Promo Frames | Stable Diffusion img2img + curated frame interpolation | Scale campaign assets while retaining product identity | Strength / Guidance / Seeds / Asset curation | **Integrated** |

## Quick Start & Service Execution

```bash
git clone https://github.com/shgupta901/RetailMind-AI.git
cd RetailMind-AI
python -m venv venv
```

Windows PowerShell activation:

```powershell
.\venv\Scripts\Activate.ps1
```

Run existing services:

```bash
python src/service.py
python src/vision_service.py
```
