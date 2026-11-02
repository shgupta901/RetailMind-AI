import os
import joblib
import pandas as pd

CHAMPION_MODEL_PATH = os.path.join("models", "churn_champion_v2.joblib")
BASELINE_MODEL_PATH = os.path.join("models", "churn_pipeline_v1.joblib")

class ChurnPredictorService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = CHAMPION_MODEL_PATH if os.path.exists(CHAMPION_MODEL_PATH) else BASELINE_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}.")

        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.version = "Day 2 Champion (Random Forest)" if "v2" in model_path else "Day 1 Baseline (Logistic Regression)"

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        preds = self.pipeline.predict(records)
        probas = self.pipeline.predict_proba(records)[:, 1]

        output = records.copy()
        output["predicted_churn"] = preds
        output["churn_risk_probability"] = probas.round(4)
        output["risk_flag"] = output["churn_risk_probability"].apply(
            lambda p: "High Risk" if p >= 0.5 else "Low Risk"
        )
        return output

if __name__ == "__main__":
    sample_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }])

    service = ChurnPredictorService()
    print(f"[SERVICE VERSION ACTIVE] {service.version}")
    result = service.predict(sample_customer)
    print(result[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])
