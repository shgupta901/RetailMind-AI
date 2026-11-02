import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "churn_pipeline_v1.joblib")

class ChurnPredictorService:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_pipeline.py first.")
        self.pipeline = joblib.load(model_path)

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
    result = service.predict(sample_customer)
    print("[SERVICE PREDICTION SUCCESS]")
    print(result[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])
