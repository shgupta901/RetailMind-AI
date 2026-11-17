"""RetailMind AI - Day 11: VAE image augmentation for rare product categories.

TensorFlow/Keras-only ML implementation. Expected dataset layout:
data/fashion_products/<category>/*.jpg

Pipeline:
1. Count categories and select the least represented class.
2. Train a convolutional VAE on that class.
3. Generate candidates and reject weak/near-duplicate outputs.
4. Train the same classifier before and after augmentation.
5. Report per-class accuracy and fairness lift.
"""

from pathlib import Path
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "data/fashion_products"
GENERATED_DIR = "generated/day11_rare_category"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
LATENT_DIM = 64
VAE_EPOCHS = 15
CLASSIFIER_EPOCHS = 8
MAX_RARE_TRAIN_IMAGES = 1500
SEED = 42


def discover_categories(data_dir=DATA_DIR):
    root = Path(data_dir)
    extensions = {".jpg", ".jpeg", ".png"}
    counts = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[folder.name] = sum(
            1 for file in folder.rglob("*") if file.suffix.lower() in extensions
        )
    counts = {name: count for name, count in counts.items() if count > 0}
    if len(counts) < 2:
        raise ValueError("Need at least two non-empty category folders.")
    rare = min(counts, key=counts.get)
    return counts, rare


def load_rare_dataset(category, data_dir=DATA_DIR):
    folder = str(Path(data_dir) / category)
    ds = keras.utils.image_dataset_from_directory(
        folder,
        labels=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    return ds.map(lambda x: tf.cast(x, tf.float32) / 255.0,
                  num_parallel_calls=tf.data.AUTOTUNE).take(
        max(1, MAX_RARE_TRAIN_IMAGES // BATCH_SIZE)
    ).prefetch(tf.data.AUTOTUNE)


def build_encoder():
    inputs = keras.Input((*IMAGE_SIZE, 3))
    x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    return keras.Model(inputs, [z_mean, z_log_var], name="day11_encoder")


def build_decoder():
    inputs = keras.Input((LATENT_DIM,))
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="day11_decoder")


@keras.utils.register_keras_serializable(package="RetailMind")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.utils.register_keras_serializable(package="RetailMind")
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=0.001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.sampling = Sampling()
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(images, training=True)
            z = self.sampling([z_mean, z_log_var])
            reconstructed = self.decoder(z, training=True)
            reconstruction = tf.reduce_mean(
                tf.reduce_sum(tf.square(images - reconstructed), axis=(1, 2, 3))
            )
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total = reconstruction + self.beta * kl
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total)
        self.reconstruction_tracker.update_state(reconstruction)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


def image_quality_mask(images, min_std=0.06, max_similarity=0.985):
    """Reject flat images and near-duplicates within the generated batch."""
    flat = tf.reshape(images, (tf.shape(images)[0], -1))
    variation = tf.math.reduce_std(flat, axis=1)
    norm = tf.math.l2_normalize(flat, axis=1)
    similarity = tf.matmul(norm, norm, transpose_b=True)
    similarity = similarity - tf.eye(tf.shape(similarity)[0]) * 2.0
    duplicate_score = tf.reduce_max(similarity, axis=1)
    return (variation >= min_std) & (duplicate_score <= max_similarity)


def generate_and_filter(decoder, needed, output_dir=GENERATED_DIR):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted = 0
    attempts = 0
    while accepted < needed and attempts < needed * 8:
        batch = min(64, max(needed - accepted, 16))
        images = decoder(tf.random.normal((batch, LATENT_DIM)), training=False)
        keep = tf.where(image_quality_mask(images))[:, 0]
        kept = tf.gather(images, keep)
        for image in kept:
            if accepted >= needed:
                break
            png = tf.io.encode_png(tf.cast(tf.clip_by_value(image, 0, 1) * 255, tf.uint8))
            tf.io.write_file(str(output / f"synthetic_{accepted:05d}.png"), png)
            accepted += 1
        attempts += batch
    return accepted, attempts


def build_classifier(class_count):
    inputs = keras.Input((*IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="day11_balanced_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def classifier_datasets(data_dir):
    train = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    )
    val = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE), train.class_names


def per_class_accuracy(model, dataset, class_names):
    correct = tf.zeros((len(class_names),), tf.float32)
    total = tf.zeros((len(class_names),), tf.float32)
    for images, labels in dataset:
        preds = tf.argmax(model(images, training=False), axis=1, output_type=labels.dtype)
        total += tf.math.bincount(labels, minlength=len(class_names), maxlength=len(class_names), dtype=tf.float32)
        matched = tf.boolean_mask(labels, tf.equal(preds, labels))
        correct += tf.math.bincount(matched, minlength=len(class_names), maxlength=len(class_names), dtype=tf.float32)
    return tf.math.divide_no_nan(correct, total)


def save_comparison(class_names, before, after, path="models/day11_per_class_accuracy.csv"):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "accuracy_before", "accuracy_after", "lift"])
        for name, old, new in zip(class_names, before.numpy(), after.numpy()):
            writer.writerow([name, float(old), float(new), float(new - old)])


def main():
    tf.random.set_seed(SEED)
    counts, rare = discover_categories()
    print("Category counts:", counts)
    print("Selected under-represented category:", rare)

    rare_ds = load_rare_dataset(rare)
    encoder, decoder = build_encoder(), build_decoder()
    vae = VAE(encoder, decoder, name="day11_rare_category_vae")
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    vae.fit(rare_ds, epochs=VAE_EPOCHS, verbose=2)

    target_count = max(counts.values())
    requested = max(0, target_count - counts[rare])
    kept, attempted = generate_and_filter(decoder, requested)
    print(f"Synthetic quality gate kept {kept} of {attempted} attempted images.")

    # Baseline classifier on original data.
    train_original, validation, class_names = classifier_datasets(DATA_DIR)
    baseline = build_classifier(len(class_names))
    baseline.fit(train_original, validation_data=validation, epochs=CLASSIFIER_EPOCHS, verbose=2)
    before = per_class_accuracy(baseline, validation, class_names)

    # Copying generated files into a dataset class folder is intentionally explicit.
    # Set AUGMENTED_DATA_DIR to a prepared copy in which generated images are placed
    # under the same rare-category folder, then train the identical classifier.
    augmented_root = Path("data/fashion_products_augmented")
    if augmented_root.exists():
        train_augmented, _, augmented_classes = classifier_datasets(str(augmented_root))
        if augmented_classes != class_names:
            raise ValueError("Original and augmented class names do not match.")
        augmented = build_classifier(len(class_names))
        augmented.fit(train_augmented, validation_data=validation, epochs=CLASSIFIER_EPOCHS, verbose=2)
        after = per_class_accuracy(augmented, validation, class_names)
        Path("models").mkdir(exist_ok=True)
        save_comparison(class_names, before, after)
        augmented.save("models/catalog_classifier_augmented_v11.keras")
        print("Saved fairness comparison and augmented classifier.")
    else:
        print("Prepare data/fashion_products_augmented to run the controlled before/after classifier comparison.")

    encoder.save("models/rare_category_encoder_v11.keras")
    decoder.save("models/rare_category_decoder_v11.keras")


if __name__ == "__main__":
    main()
