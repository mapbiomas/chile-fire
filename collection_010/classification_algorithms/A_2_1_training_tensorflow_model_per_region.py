# ==========================================
# A_2_1_training_tensorflow_model_per_region.py
# TensorFlow 2.x rewrite – faithful to original
# ==========================================

import os
import gc
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from A_0_2_log_algorithm_monitor import log
from google.cloud import storage


# -----------------------------
# Global configuration
# -----------------------------
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MODEL_ROOT_DIR = "./models"
METRICS_FILE = "training_metrics.json"


# -----------------------------
# Model builder
# -----------------------------
def build_model(input_shape, num_classes=2):
    """Builds a simple fully connected network (same architecture as TF1 version)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# GCS upload helper
# -----------------------------
def upload_to_gcs(bucket_name, local_path, gcs_path):
    """Uploads a file to GCS using the official client."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        log.log_message(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{gcs_path}", stage="upload")
    except Exception as e:
        log.log_message(f"⚠️ Failed to upload {local_path}: {e}", stage="upload", level="error")


# -----------------------------
# Save metrics to JSON
# -----------------------------
def save_metrics(save_dir, history):
    """Saves training metrics (loss, accuracy, etc.) to JSON."""
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    metrics_path = os.path.join(save_dir, METRICS_FILE)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.log_message(f"📊 Metrics saved at {metrics_path}", stage="training")


# -----------------------------
# Training function (per region)
# -----------------------------
def train_model_for_region(region_id, x_train, y_train, x_val=None, y_val=None,
                           country="BRA", bucket_name=None):
    """Trains, saves and uploads a model for a specific region."""
    start_time = time.time()
    log.log_message(f"🧠 Starting training for region {region_id}", stage="training")
    log.resources()

    try:
        input_shape = (x_train.shape[1],)
        model = build_model(input_shape=input_shape)
        log.log_message(f"Model built successfully for region {region_id}", stage="training")

        # Define callbacks
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        checkpoint_dir = os.path.join(MODEL_ROOT_DIR, country, f"region_{region_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
        checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")

        # Train model
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, checkpoint_cb],
            verbose=1
        )

        duration_min = round((time.time() - start_time) / 60, 2)
        log.log_message(f"✅ Training completed for region {region_id} in {duration_min} min", stage="training")

        # Save model and metrics
        model_path = os.path.join(checkpoint_dir, "final_model.h5")
        model.save(model_path)
        save_metrics(checkpoint_dir, history)

        # Upload if bucket provided
        if bucket_name:
            upload_to_gcs(bucket_name, model_path, f"{country}/region_{region_id}/final_model.h5")
            upload_to_gcs(bucket_name, os.path.join(checkpoint_dir, METRICS_FILE),
                          f"{country}/region_{region_id}/{METRICS_FILE}")

    except Exception as e:
        log.log_message(f"❌ Error during training for region {region_id}: {e}", stage="training", level="error")
        raise

    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        log.resources()
        log.log_message(f"🏁 Training session cleared for region {region_id}", stage="training")


# -----------------------------
# Multi-region orchestrator
# -----------------------------
def run_training(regions_data, country="BRA", bucket_name=None):
    """
    Runs training for multiple regions sequentially.
    regions_data: dict {region_id: (x_train, y_train, x_val, y_val)}
    """
    log.log_message(f"🚀 Starting model training for {len(regions_data)} regions", stage="training")

    for region_id, data in regions_data.items():
        try:
            if len(data) == 2:
                x_train, y_train = data
                x_val, y_val = None, None
            else:
                x_train, y_train, x_val, y_val = data
            train_model_for_region(region_id, x_train, y_train, x_val, y_val, country, bucket_name)
        except Exception as e:
            log.log_message(f"⚠️ Training failed for region {region_id}: {e}", stage="training", level="error")

    log.log_message("✅ All regional training tasks completed", stage="training")
    log.summary("completed")


# -----------------------------
# Test entry point
# -----------------------------
if __name__ == "__main__":
    x_train = np.random.rand(500, 10)
    y_train = np.random.randint(0, 2, 500)

    regions_data = {
        "r01": (x_train, y_train),
        "r02": (x_train, y_train)
    }

    run_training(regions_data, country="BRA", bucket_name=None)
