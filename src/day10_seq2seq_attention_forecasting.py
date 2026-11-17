"""RetailMind AI - Day 10: Seq2Seq demand forecasting with attention.

Uses TensorFlow/Keras and Python standard-library modules only.
Expected Rossmann CSV columns: Store, Date, Sales. The first store is used unless
STORE_ID is changed. The model maps 56 historical daily sales values to a
14-day forecast and exposes attention weights for explanation.
"""

from pathlib import Path
import csv
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CSV_PATH = "train.csv"
STORE_ID = None
LOOKBACK = 56
HORIZON = 14
TEST_DAYS = 56
VAL_DAYS = 56
EPOCHS = 15
BATCH_SIZE = 64
MODEL_PATH = "models/demand_seq2seq_attention_v10.keras"
FORECAST_PATH = "models/day10_multihorizon_forecast.csv"
ATTENTION_CSV_PATH = "models/day10_attention_weights.csv"
ATTENTION_IMAGE_PATH = "models/day10_attention_heatmap.png"


def load_daily_sales(csv_path=CSV_PATH, store_id=STORE_ID):
    """Read Rossmann CSV and return one continuous store-level daily series."""
    rows = []
    selected_store = store_id
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Date", "Sales"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            if selected_store is None and "Store" in row:
                selected_store = row["Store"]
            if "Store" in row and selected_store is not None and row["Store"] != str(selected_store):
                continue
            rows.append((datetime.strptime(row["Date"], "%Y-%m-%d").date(), float(row["Sales"])))

    if not rows:
        raise ValueError("No sales rows were found for the selected store.")

    by_date = {}
    for date, sales in rows:
        by_date[date] = by_date.get(date, 0.0) + sales

    start, end = min(by_date), max(by_date)
    dates, sales = [], []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        sales.append(by_date.get(current, 0.0))
        current += timedelta(days=1)

    return dates, tf.constant(sales, dtype=tf.float32), selected_store


def make_windows(series, lookback=LOOKBACK, horizon=HORIZON):
    """Create history -> multi-horizon targets with chronological overlap."""
    total = lookback + horizon
    windows = tf.signal.frame(series, frame_length=total, frame_step=1)
    x = windows[:, :lookback]
    y = windows[:, lookback:]
    return x[..., tf.newaxis], y


def prepare_data(series):
    """Chronological train/validation/test split with train-only normalization."""
    test_start = len(series) - TEST_DAYS
    val_start = test_start - VAL_DAYS
    train_series = series[:val_start]

    mean = tf.reduce_mean(train_series)
    std = tf.math.reduce_std(train_series)
    std = tf.where(std > 0, std, tf.constant(1.0, tf.float32))
    normalized = (series - mean) / std

    x_all, y_all = make_windows(normalized)
    target_end_indices = tf.range(LOOKBACK + HORIZON - 1, len(series))
    train_mask = target_end_indices < val_start
    val_mask = (target_end_indices >= val_start) & (target_end_indices < test_start)
    test_mask = target_end_indices >= test_start

    return (
        tf.boolean_mask(x_all, train_mask), tf.boolean_mask(y_all, train_mask),
        tf.boolean_mask(x_all, val_mask), tf.boolean_mask(y_all, val_mask),
        tf.boolean_mask(x_all, test_mask), tf.boolean_mask(y_all, test_mask),
        mean, std,
    )


@keras.utils.register_keras_serializable(package="RetailMind")
class TemporalAttention(layers.Layer):
    """Additive attention from decoder states to all encoder states."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query_projection = layers.Dense(units, use_bias=False)
        self.value_projection = layers.Dense(units, use_bias=False)
        self.score_projection = layers.Dense(1, use_bias=False)

    def call(self, inputs):
        decoder_states, encoder_states = inputs
        query = self.query_projection(decoder_states)[:, :, tf.newaxis, :]
        values = self.value_projection(encoder_states)[:, tf.newaxis, :, :]
        scores = self.score_projection(tf.nn.tanh(query + values))
        scores = tf.squeeze(scores, axis=-1)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, encoder_states)
        return context, weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_seq2seq_attention():
    """Encoder-decoder LSTM with attention and two outputs."""
    encoder_input = keras.Input((LOOKBACK, 1), name="history")
    encoder_states, state_h, state_c = layers.LSTM(
        64, return_sequences=True, return_state=True, name="encoder_lstm"
    )(encoder_input)

    repeated_context = layers.RepeatVector(HORIZON, name="repeat_context")(state_h)
    decoder_states = layers.LSTM(
        64, return_sequences=True, name="decoder_lstm"
    )(repeated_context, initial_state=[state_h, state_c])

    context, attention_weights = TemporalAttention(64, name="temporal_attention")(
        [decoder_states, encoder_states]
    )
    merged = layers.Concatenate(name="decoder_context")([decoder_states, context])
    forecast = layers.TimeDistributed(layers.Dense(1), name="forecast_steps")(merged)
    forecast = layers.Reshape((HORIZON,), name="forecast")(forecast)

    training_model = keras.Model(encoder_input, forecast, name="seq2seq_attention_forecaster")
    explanation_model = keras.Model(
        encoder_input, [forecast, attention_weights], name="seq2seq_attention_explainer"
    )
    training_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return training_model, explanation_model


def metrics(actual, predicted):
    error = actual - predicted
    mae = tf.reduce_mean(tf.abs(error))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(error)))
    mape = tf.reduce_mean(tf.abs(error) / tf.maximum(tf.abs(actual), 1.0)) * 100.0
    return float(mae), float(rmse), float(mape)


def save_attention_heatmap(weights, path=ATTENTION_IMAGE_PATH):
    """Save a TensorFlow-only grayscale attention heatmap PNG."""
    image = weights[tf.newaxis, ..., tf.newaxis]
    image = tf.image.resize(image, [HORIZON * 20, LOOKBACK * 6], method="nearest")
    image = tf.cast(tf.clip_by_value(image / tf.reduce_max(image), 0, 1) * 255, tf.uint8)
    tf.io.write_file(path, tf.io.encode_png(image[0]))


def main():
    tf.random.set_seed(42)
    dates, series, store = load_daily_sales()
    data = prepare_data(series)
    x_train, y_train, x_val, y_val, x_test, y_test, mean, std = data

    model, explainer = build_seq2seq_attention()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    pred_norm, attention = explainer.predict(x_test, verbose=0)
    actual = y_test * std + mean
    predicted = pred_norm * std + mean
    mae, rmse, mape = metrics(actual, predicted)

    # Day 9-style single-step persistence baseline repeated across the horizon.
    baseline = tf.repeat((x_test[:, -1, 0] * std + mean)[:, tf.newaxis], HORIZON, axis=1)
    b_mae, b_rmse, b_mape = metrics(actual, baseline)

    print(f"Store: {store}; lookback={LOOKBACK}; horizon={HORIZON}")
    print(f"Persistence baseline: MAE={b_mae:.2f}, RMSE={b_rmse:.2f}, MAPE={b_mape:.2f}%")
    print(f"Seq2Seq + attention: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

    Path("models").mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    with open(FORECAST_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["horizon_day", "actual_sales", "predicted_sales"])
        for step in range(HORIZON):
            writer.writerow([step + 1, float(actual[0, step]), float(predicted[0, step])])

    first_attention = attention[0]
    with open(ATTENTION_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["forecast_step"] + [f"history_day_{i-LOOKBACK+1}" for i in range(LOOKBACK)])
        for step in range(HORIZON):
            writer.writerow([step + 1] + first_attention[step].tolist())
    save_attention_heatmap(first_attention)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved forecast: {FORECAST_PATH}")
    print(f"Saved attention weights: {ATTENTION_CSV_PATH}")
    print(f"Saved attention heatmap: {ATTENTION_IMAGE_PATH}")


if __name__ == "__main__":
    main()
