# Day 10 - Seq2Seq and Attention Demand Forecasting

## Purpose
Day 10 extends RetailMind AI's Day 9 next-day LSTM into a multi-horizon demand service. The model consumes 56 daily sales observations and predicts the following 14 days, giving procurement teams a two-week planning horizon.

## Business rationale
A single next-day prediction is useful for immediate operations but is insufficient for supplier lead times, promotion planning, and distribution. Multi-step forecasts expose the expected demand path across several future periods. Attention adds an interpretable matrix showing which historical days the model emphasized for each forecast step.

## Data contract
The pipeline expects Rossmann `train.csv` in the repository root.

Required columns:
- `Date`: ISO date in `YYYY-MM-DD` format.
- `Sales`: numeric daily demand.

Optional column:
- `Store`: when present, the loader selects the first store unless `STORE_ID` is configured.

Missing calendar dates are filled with zero sales. For production, store-closure information should be considered before treating all zero-demand days as missing observations.

## Windowing and split design
- Look-back: 56 days.
- Forecast horizon: 14 days.
- Test period: latest 56 target days.
- Validation period: preceding 56 target days.
- Training period: all earlier targets.
- Normalization: mean and standard deviation fitted on training data only.

This chronological design prevents future values or future normalization statistics from leaking into training.

## Model architecture
1. A 64-unit encoder LSTM reads the complete 56-day history and returns every hidden state.
2. The final encoder state initializes a 64-unit decoder LSTM.
3. The decoder emits 14 states, one for each future day.
4. Additive temporal attention compares every decoder state with all encoder states.
5. Softmax attention weights create a weighted history context for each forecast day.
6. A time-distributed Dense layer emits the 14 normalized demand predictions.

## Evaluation
The model is compared with a persistence baseline that repeats the final observed demand for all 14 future steps. Both are scored in original sales units using:
- MAE: average absolute forecasting error.
- RMSE: penalizes large misses more strongly.
- MAPE: percentage error with denominator protection for zero demand.

The comparison is more meaningful than claiming success from training loss alone. Test metrics should be evaluated once after model development.

## Attention explanation
The explainer model returns a `(14, 56)` matrix for each sample. Rows correspond to forecast days and columns to history days. Each row sums to one. The pipeline exports:
- `models/day10_attention_weights.csv`
- `models/day10_attention_heatmap.png`

The heatmap is generated with TensorFlow only. Bright cells represent higher relative attention. Attention describes model relevance rather than causal business impact.

## Generated artifacts
After successful execution:
- `models/demand_seq2seq_attention_v10.keras`
- `models/day10_multihorizon_forecast.csv`
- `models/day10_attention_weights.csv`
- `models/day10_attention_heatmap.png`

## Run
```bash
python3 src/day10_seq2seq_attention_forecasting.py
```

TensorFlow must be installed and `train.csv` must exist in the repository root.

## Procurement integration
The 14-day forecast CSV can feed reorder-point or safety-stock calculations. A production version should add promotions, holidays, store-open flags, product identifiers, supplier lead times, and prediction intervals. Rolling-origin backtesting should also replace dependence on a single final test period.
