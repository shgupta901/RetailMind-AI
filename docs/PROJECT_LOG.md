# RetailMind AI — Daily Development Log

## Day 1: End-to-End Baseline Pipeline & ML-as-a-Service
- **Changes Added**:
  - Initialized repo structure (data/, src/, models/, notebooks/, docs/).
  - Built scikit-learn Pipeline with ColumnTransformer (scaling + one-hot encoding).
  - Trained baseline Logistic Regression model and exported artefact via joblib.
  - Implemented service wrapper (src/service.py) for inference.
- **Business Value Added**:
  - Provides immediate baseline churn risk scoring for retail account teams.
- **Open Blockers / Points for Doubt-Clearing**:
  - Recall on churn is 53%, missing at-risk users due to class imbalance.
