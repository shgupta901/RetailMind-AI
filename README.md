# RetailMind AI

RetailMind AI is an enterprise-grade retail intelligence platform developed incrementally across 35 production milestones.

## Increments & Model Architecture

### Day 1: Churn ML Backbone & Baseline Service
- End-to-end scikit-learn pipeline (Cleaning -> ColumnTransformer -> Logistic Regression -> joblib persistence).

### Day 2: Tree-Based & Margin-Based Models (Recall-Driven Evaluation)
- Benchmarked Decision Tree, Random Forest, and SVM against Day 1's baseline.
- Objective: Capture non-linear relationships and maximize recall on the positive churn class (catch churners before they leave).

#### Model Comparison Table
| Model | Accuracy | Recall (Churn=1) | Precision (Churn=1) | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Day 1)** | 0.8053 | 0.5722 | 0.6544 | 0.8361 | Deprecated Baseline |
| **Decision Tree (depth=6)** | 0.7328 | 0.7487 | 0.4965 | 0.8092 | Evaluated |
| **SVM (RBF Kernel)** | 0.7647 | 0.7594 | 0.5408 | 0.8286 | Runner-Up |
| **Random Forest (150 trees)** | **0.7854** | **0.7567** | **0.5739** | **0.8491** | **Champion (Day 2)** |

#### Why Random Forest Won
- **Recall Jump**: Increased from 57.2% to 75.7% (+18.5 percentage points), identifying 3 out of 4 churning customers.
- **Precision Advantage**: At 57.4%, Random Forest produced substantially fewer false alarms than Decision Tree (49.6%).
- **Superior Discriminative Power**: Delivered the highest overall ROC-AUC (0.8491).
