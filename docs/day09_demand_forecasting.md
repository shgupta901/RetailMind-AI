# Day 9: RNN / LSTM Demand Forecasting

## Objective
Forecast daily store/product demand with an LSTM so RetailMind AI can support inventory-planning decisions and reduce stockouts and overstock.

## Dataset
The implementation expects `train.csv` from the Rossmann Store Sales dataset. Required columns are `Date` and `Sales`; when `Store` exists, the first store is selected to produce one clean daily series.

## Method
1. Parse dates and aggregate sales by day with pandas.
2. Resample to a continuous daily frequency.
3. Keep the latest 42 days as an untouched test period.
4. Fit min-max scaling values on the training period only.
5. Convert the series into 30-day look-back windows predicting the next day.
6. Train a Keras LSTM with 64 units and a small Dense prediction head.
7. Evaluate in original sales units using MAE and RMSE.
8. Save the trained forecaster and actual-versus-predicted results.

## Project files
- `src/day09_lstm_demand_forecasting.py`: reusable training pipeline.
- `notebooks/day09_lstm_demand_forecasting.ipynb`: notebook entry point and project explanation.
- `models/demand_lstm_v9.keras`: generated after successful training.
- `models/day09_forecast_results.csv`: generated evaluation predictions.

## Run
From the repository root:

```bash
python src/day09_lstm_demand_forecasting.py
```

TensorFlow must be installed and `train.csv` must be in the repository root. Model artifacts are intentionally generated at runtime and should not be claimed until training succeeds.
