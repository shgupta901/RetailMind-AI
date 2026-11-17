"""RetailMind AI - Day 9: LSTM demand forecasting.

Expected input: train.csv from Rossmann Store Sales with Date, Sales and
optionally Store columns. The module prepares a daily time series, creates
look-back windows, holds out the latest 42 days, trains an LSTM, reports MAE
and RMSE, writes actual-vs-predicted values, and saves the Keras model.
"""

from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras

CSV_PATH = "train.csv"
LOOK_BACK = 30
TEST_DAYS = 42
EPOCHS = 10
BATCH_SIZE = 32
MODEL_PATH = "models/demand_lstm_v9.keras"
RESULTS_PATH = "models/day09_forecast_results.csv"


def load_sales_series(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    required = {"Date", "Sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    if "Store" in df.columns:
        store_id = df["Store"].iloc[0]
        df = df[df["Store"] == store_id].copy()
        print(f"Using Store={store_id}")

    series = (
        df[["Date", "Sales"]]
        .dropna()
        .groupby("Date", as_index=False)["Sales"].sum()
        .sort_values("Date")
        .set_index("Date")["Sales"]
        .resample("D").sum()
        .astype("float32")
    )
    if len(series) <= LOOK_BACK + TEST_DAYS:
        raise ValueError("Not enough daily observations for look-back and test periods.")
    return series


def make_windows(values, look_back):
    x, y = [], []
    values = values.tolist()
    for i in range(len(values) - look_back):
        x.append(values[i:i + look_back])
        y.append(values[i + look_back])
    return (
        tf.constant(x, dtype=tf.float32)[..., tf.newaxis],
        tf.constant(y, dtype=tf.float32),
    )


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(LOOK_BACK, 1)),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    tf.random.set_seed(42)
    series = load_sales_series()

    train_values = series.iloc[:-TEST_DAYS]
    min_sales = float(train_values.min())
    max_sales = float(train_values.max())
    scale = max(max_sales - min_sales, 1.0)
    scaled = (series - min_sales) / scale

    train_scaled = scaled.iloc[:-TEST_DAYS]
    test_scaled = scaled.iloc[-(TEST_DAYS + LOOK_BACK):]
    x_train, y_train = make_windows(train_scaled, LOOK_BACK)
    x_test, y_test = make_windows(test_scaled, LOOK_BACK)

    model = build_model()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )

    pred_scaled = tf.squeeze(model.predict(x_test, verbose=0), axis=-1)
    predictions = pred_scaled * scale + min_sales
    actual = y_test * scale + min_sales
    mae = tf.reduce_mean(tf.abs(actual - predictions))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(actual - predictions)))

    print(f"Test MAE: {float(mae):.2f}")
    print(f"Test RMSE: {float(rmse):.2f}")

    comparison = pd.DataFrame({
        "actual_sales": actual.numpy(),
        "predicted_sales": predictions.numpy(),
    })
    print("\nPredicted vs actual sample:")
    print(comparison.head(10))

    Path("models").mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    comparison.to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
