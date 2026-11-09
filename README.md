# RetailMind AI — Enterprise Retail Intelligence Platform

RetailMind AI is an end-to-end retail intelligence platform developed incrementally across 35 production milestones. It unifies predictive churn analytics, deep learning propensity scoring, computer vision image classification, object detection, background segmentation, and explainable AI.

---

## ??? Phase 1 & Phase 2 Master Milestone Ledger

### Part A: Phase 1 — Classical ML & Deep Churn Engine (Days 01–05)
| Day | Focus / Architecture | Accuracy | Recall (Churn=1) | Precision (Churn=1) | ROC-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **01** | Baseline Logistic Regression Pipeline | 0.8053 | 0.5722 | 0.6544 | 0.8361 | Deprecated |
| **02** | Decision Tree / Random Forest / SVM | 0.7854 | 0.7567 | 0.5739 | 0.8491 | Evaluated |
| **03** | Tuned Random Forest (5-Fold CV) | 0.7925 | 0.7701 | 0.5842 | 0.8498 | Audited |
| **04** | Feature Eng + XGBoost Ensemble | **0.8010** | **0.8048** | **0.5960** | **0.8612** | **Production Champion** |
| **05** | Deep ANN (Keras + BatchNorm + Dropout) | 0.7583 | 0.7834 | 0.5288 | 0.8415 | DL Benchmark |

### Part B: Phase 2 — Computer Vision & Catalog Intelligence (Days 06–08)
| Day | Vision Module | Target Architecture | Accuracy / Score | Primary Business Value | Status |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **06** | Product Auto-Tagger | Sequential CNN (Conv2D + MaxPool) | ~0.90 | Automates manual catalog data entry | **V1 Shipped** |
| **07** | Transfer Learning & Detection | MobileNetV2 Backbone + YOLOv8 | ~0.94 | High-res RGB classification & multi-item QA | **Shipped** |
| **08** | Packaged CV Module | OpenCV GrabCut + CNN + Grad-CAM | ~0.90 | End-to-end background removal & explainable AI | **Phase 2 Complete** |

---

## ? Quick Start & Service Execution

`ash
# Clone and setup
git clone [https://github.com/shgupta901/RetailMind-AI.git](https://github.com/shgupta901/RetailMind-AI.git)
cd RetailMind-AI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run tabular inference or vision checks
python src/service.py
python src/vision_service.py
