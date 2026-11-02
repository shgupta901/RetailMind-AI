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

## Day 2: Decision Tree, Random Forest & SVM
- **Changes Added**:
  - Implemented multi-model comparative training harness (src/pipelines/train_comparative_models.py).
  - Evaluated Decision Tree, Random Forest, and SVM models on identical preprocessed features.
  - Selected Random Forest as Champion and persisted to models/churn_champion_v2.joblib.
  - Updated src/service.py to auto-resolve and serve Champion v2.
  - Added comparison table in README.md and created day02 notebook.
- **Business Value Added**:
  - Retention teams now capture ~75.7% of all churning accounts (up from 57.2%), preserving recurring revenue by triggering proactive retention outreach.
- **Open Blockers / Points for Doubt-Clearing**:
  - SVM training runtime is noticeably higher due to quadratic scaling; evaluate whether hyperparameter search or gradient boosting offers a better speed-to-recall trade-off.
