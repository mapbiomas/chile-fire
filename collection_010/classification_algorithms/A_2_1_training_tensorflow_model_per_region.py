# ==========================================
# A_2_1_training_tensorflow_model_per_region.py
# TensorFlow 2.x – Compatible rewrite (MapBiomas Fire pipeline)
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
# Global training configuration
# -----------------------------
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MODEL_NAME = "model_final.h5"
METRICS_FILE = "training_metrics.json"


# -----------------------------
# Helper: build model
# -----------------------------
def build_model(input_shape, num_classes=2):
    """Builds a simple feedforward network for burned area classification."""
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
# Helper: upload file to GCS
# -----------------------------
def upload_to_gcs(bucket_name, local_path, gcs_path):
    """Uploads a local file to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        log.log_message(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{gcs_path}", stage="upload")
    except Exception as e:
        log.log_message(f"⚠️ Upload failed for {local_path}: {e}", stage="upload", level="error")


# -----------------------------
# Helper: save metrics to JSON
# -----------------------------
def save_metrics(save_dir, history):
    """Saves loss/accuracy metrics to JSON."""
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    metrics_path = os.path.join(save_dir, METRICS_FILE)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.log_message(f"📊 Metrics saved at {metrics_path}", stage="training")


# -----------------------------
# Training function per region
# -----------------------------
def train_model_for_region(region_id, x_train, y_train,
                           country, base_dataset_path, bucket_name,
                           overwrite=False):
    """Trains and saves a TensorFlow model for a given region."""
    start_time = time.time()
    log.log_message(f"🧠 Starting training for region {region_id}", stage="training")
    log.resources()

    try:
        # Define save directory (compatible with original structure)
        save_dir = os.path.join(base_dataset_path, "models_col1")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, MODEL_NAME)
        metrics_path = os.path.join(save_dir, METRICS_FILE)

        # Handle existing model
        if os.path.exists(model_path):
            if overwrite:
                log.log_message(f"⚠️ Existing model found. Overwriting...", stage="training")
                os.remove(model_path)
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)
            else:
                log.log_message(f"⏭️ Model already exists. Skipping training (overwrite=False).", stage="training")
                return

        # Build and train
        input_shape = (x_train.shape[1],)
        model = build_model(input_shape=input_shape)
        log.log_message("✅ Model built successfully", stage="training")

        # Callbacks
        early_stop = callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
        checkpoint_cb = callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="loss")

        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, checkpoint_cb],
            verbose=1
        )

        duration_min = round((time.time() - start_time) / 60, 2)
        log.log_message(f"✅ Training completed for region {region_id} in {duration_min} min", stage="training")

        # Save final model and metrics
        model.save(model_path)
        save_metrics(save_dir, history)

        # Upload to GCS (same structure as original pipeline)
        if bucket_name:
            relative_path = f"{base_dataset_path}/models_col1"
            upload_to_gcs(bucket_name, model_path, f"{relative_path}/{MODEL_NAME}")
            upload_to_gcs(bucket_name, metrics_path, f"{relative_path}/{METRICS_FILE}")

    except Exception as e:
        log.log_message(f"❌ Error during training for region {region_id}: {e}", stage="training", level="error")
        raise

    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        log.resources()
        log.log_message(f"🏁 Training session cleared for region {region_id}", stage="training")


# -----------------------------
# Orchestrator for multi-region
# -----------------------------
def run_training(regions_data, country, bucket_name, base_dataset_path, overwrite=False):
    """
    Trains models for all selected regions sequentially.
    regions_data: dict {region_id: (x_train, y_train)}
    """
    log.log_message(f"🚀 Starting model training for {len(regions_data)} regions", stage="training")

    for region_id, data in regions_data.items():
        try:
            x_train, y_train = data
            train_model_for_region(
                region_id, x_train, y_train,
                country=country,
                base_dataset_path=base_dataset_path,
                bucket_name=bucket_name,
                overwrite=overwrite
            )
        except Exception as e:
            log.log_message(f"⚠️ Training failed for region {region_id}: {e}", stage="training", level="error")

    log.log_message("✅ All regional training tasks completed", stage="training")
    log.summary("completed")
