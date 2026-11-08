import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "models"
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "catalog_cnn_v6.keras")

def run_day6_cnn():
    print("[INFO] Loading Fashion-MNIST for Retail Catalog Prototyping...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Reshape (28, 28) to (28, 28, 1) to represent grayscale channel and normalize
    train_images = np.expand_dims(train_images, -1).astype("float32") / 255.0
    test_images = np.expand_dims(test_images, -1).astype("float32") / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"[INFO] Training data shape: {train_images.shape}")

    # Build CNN Architecture with Native Augmentation
    model = Sequential([
        # Data Augmentation layer (runs only during training)
        RandomFlip("horizontal", input_shape=(28, 28, 1)),
        RandomRotation(0.05),

        # Block 1: Feature Extraction
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2: Deep Feature Extraction
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3: Classification Head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("[INFO] Compiling CNN and starting training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        train_images, train_labels,
        epochs=20,
        validation_split=0.2,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    print("\n[INFO] Evaluating CNN on Test Set...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 50)
    print("DAY 6 CNN CATALOG CLASSIFIER RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:\n", classification_report(test_labels, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print(confusion_matrix(test_labels, y_pred))
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(CNN_MODEL_PATH)
    print(f"\n[SUCCESS] CNN Vision Module saved to: {CNN_MODEL_PATH}")

if __name__ == "__main__":
    run_day6_cnn()
