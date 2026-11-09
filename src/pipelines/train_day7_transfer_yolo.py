import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from ultralytics import YOLO

MODEL_DIR = "models"
TRANSFER_MODEL_PATH = os.path.join(MODEL_DIR, "catalog_transfer_v7.keras")

def build_transfer_classifier(input_shape=(64, 64, 3), num_classes=10):
    # Pre-trained backbone with frozen weights
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model

def run_day7_pipeline():
    print("[INFO] Building Day 7 Transfer Learning Architecture (MobileNetV2 backbone)...")
    model, base_model = build_transfer_classifier()

    # Synthetic catalog batch for verification
    X_sample = np.random.randint(0, 256, (64, 64, 64, 3)).astype("float32")
    y_sample = np.random.randint(0, 10, 64)

    # Stage 1: Train classification head
    print("[INFO] Training top classification head...")
    model.fit(X_sample, y_sample, epochs=2, batch_size=16, verbose=1)

    # Stage 2: Fine-tune top layers of the backbone
    print("[INFO] Unfreezing top 20 backbone layers for fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_sample, y_sample, epochs=1, batch_size=16, verbose=1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(TRANSFER_MODEL_PATH)
    print(f"[SUCCESS] Transfer Learning Model saved to: {TRANSFER_MODEL_PATH}")

    # Stage 3: Object Detection Prototype (YOLOv8)
    print("\n[INFO] Loading Ultralytics YOLOv8 for catalog bounding box detection...")
    yolo_detector = YOLO("yolov8n.pt")  # Auto-downloads official nano weights
    print("[SUCCESS] YOLOv8 initialized for catalog QA and multi-product shelf localization.")

if __name__ == "__main__":
    run_day7_pipeline()
