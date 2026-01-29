# ==========================================
# A_2_1_training_tensorflow_model_per_region.py
# TensorFlow 2.x compatible training module
# Fully aligned with MapBiomas-Fire pipeline structure
# ==========================================

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import subprocess
from datetime import datetime

# --- ensure eager execution for tf.data pipelines ---
tf.data.experimental.enable_debug_mode()

# --- Safe import for Colab runtime resets ---
try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None and "country" in ipy.user_ns:
        country = ipy.user_ns["country"]
        algorithms = f"/content/{country}-fire/collection_010/classification_algorithms"
        if algorithms not in sys.path:
            sys.path.append(algorithms)
except Exception:
    pass

from A_0_2_log_algorithm_monitor import log


# ======================================================
# UTILS
# ======================================================
def ensure_dir(path):
    """Create local directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def upload_to_gcs(local_path, bucket_path):
    """Upload file or folder to GCS using gsutil."""
    try:
        subprocess.run(["gsutil", "-m", "cp", "-r", local_path, bucket_path], check=True)
        log.log_message(f"☁️ Uploaded {local_path} → {bucket_path}", stage="train")
    except subprocess.CalledProcessError as e:
        log.log_message(f"⚠️ Failed to upload {local_path}: {e}", stage="train", level="error")


def build_model(input_shape):
    """Builds a simple neural network compatible with TFv1 and TFv2."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======================================================
# MAIN TRAINING FUNCTION
# ======================================================
def train_region_model(model_name, data_tuple, country, base_dataset_path, bucket_name):
    """Train and upload model for a single region."""
    start_time = time.time()

    log.log_message(f"🧠 Starting training for {model_name}", stage="train")
    log.resources()

    x_train, y_train = data_tuple
    model_dir = f"./models_col1"
    ensure_dir(model_dir)

    # Build and train model
    model = build_model(input_shape=(x_train.shape[1],))
    checkpoint_path = os.path.join(model_dir, f"{model_name}.h5")

    # Train model (verbose for monitoring)
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Save model locally
    model.save(checkpoint_path)
    log.log_message(f"💾 Model saved locally at {checkpoint_path}", stage="train")

    # Save metrics
    eval_loss, eval_acc = model.evaluate(x_train, y_train, verbose=0)
    metrics = {
        "accuracy": float(eval_acc),
        "loss": float(eval_loss),
        "timestamp": datetime.utcnow().isoformat()
    }

    metrics_path = os.path.join(model_dir, f"{model_name}_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.log_message(f"📊 Metrics saved at {metrics_path}", stage="train")

    # Upload to GCS
    gcs_model_path = f"gs://{base_dataset_path}/models_col1/{model_name}.h5"
    gcs_metrics_path = f"gs://{base_dataset_path}/models_col1/{model_name}_training_metrics.json"

    upload_to_gcs(checkpoint_path, gcs_model_path)
    upload_to_gcs(metrics_path, gcs_metrics_path)

    log.resources()
    log.log_message(f"🏁 Training session cleared for {model_name}", stage="train")

    elapsed = (time.time() - start_time) / 60.0
    log.log_message(f"✅ Training completed for {model_name} in {elapsed:.2f} min", stage="train")


# ======================================================
# ENTRYPOINT (called by A_2_0)
# ======================================================
def run_training(regions_data, country, bucket_name, base_dataset_path, overwrite=True):
    """
    Receives dict: { model_name: (x_train, y_train), ... }
    Trains and uploads each model.
    """
    log.log_message("🚀 Starting model training for all selected regions", stage="train")
    log.log_message(f"🔹 Total regions: {len(regions_data)}", stage="train")

    for model_name, data_tuple in regions_data.items():
        try:
            train_region_model(model_name, data_tuple, country, base_dataset_path, bucket_name)
        except Exception as e:
            log.log_message(f"❌ ❌ Error training {model_name}: {e}", stage="train", level="error")

    log.log_message("✅ All regional training tasks completed", stage="train")
    log.summary("completed")
