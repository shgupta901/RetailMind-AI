# RetailMind AI

RetailMind AI is an enterprise-grade retail intelligence platform developed incrementally across 35 production milestones.

## Day 1: Churn ML Backbone & Baseline Service
- **Objective**: Establish the core ML lifecycle (Ingestion -> Preprocessing -> Training -> Persistence -> Service interface).
- **Dataset**: Telco Customer Churn (Kaggle).
- **Stack**: Python, pandas, scikit-learn, joblib.

### Baseline Performance Metrics
| Metric | Baseline Score (Day 1) |
| :--- | :--- |
| **Model** | Logistic Regression (StandardScaler + OneHotEncoder) |
| **Accuracy** | ~0.8038 |
| **ROC-AUC** | ~0.8445 |
| **Precision (Churn=1)** | ~0.66 |
| **Recall (Churn=1)** | ~0.53 |
