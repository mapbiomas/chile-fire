# ==========================================
# A_2_1_training_tensorflow_model_per_region.py
# TensorFlow 2.x – Training per region with logging, compatible with original pipeline
# ==========================================

# --- Handle imports safely for Colab runtime resets ---
import os
import sys
import json
import time
import subprocess
import numpy as np
import tensorflow as tf

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

# --- Local imports (work even after reset) ---
from A_0_2_log_algorithm_monitor import log


def build_model(input_shape):
    """Builds a simple LSTM model (same as original)."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def run_training(regions_data, country, bucket_name, base_dataset_path, overwrite=True):
    """
    Trains TensorFlow models for each region, uploads results to GCS,
    and logs progress using A_0_2_log_algorithm_monitor.
    """

    log.log_message("🚀 Starting model training process", stage="training")

    if not regions_data:
        log.log_message("⚠️ No training data provided. Exiting.", stage="training", level="warning")
        return

    os.makedirs("./models", exist_ok=True)
    local_model_dir = f"./models/{country}"
    os.makedirs(local_model_dir, exist_ok=True)

    start_time_all = time.time()

    for region, (x_train, y_train) in regions_data.items():
        start_time = time.time()
        region_label = region.replace("r", "region_")

        log.log_message(f"🧠 Starting training for region {region}", stage="training")
        log.resources()

        # --- Model building ---
        model = build_model((x_train.shape[1], 1))
        model.summary(print_fn=lambda x: log.log_message(x, stage="training"))

        # --- Reshape input if needed ---
        if len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1)

        # --- Training parameters ---
        EPOCHS = 50
        BATCH_SIZE = 64
        EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        # --- Train model ---
        history = model.fit(
            x_train, y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[EARLY_STOPPING]
        )

        # --- Save model locally ---
        local_region_dir = os.path.join(local_model_dir, region_label)
        os.makedirs(local_region_dir, exist_ok=True)

        model_file = os.path.join(local_region_dir, "final_model.keras")
        model.save(model_file)

        # --- Save training metrics ---
        metrics = {
            "region": region,
            "epochs": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "duration_min": round((time.time() - start_time) / 60, 2),
        }

        metrics_file = os.path.join(local_region_dir, "training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        log.log_message(f"✅ Training completed for region {region} in {metrics['duration_min']} min", stage="training")

        # --- Upload to GCS ---
        gcs_model_path = f"gs://{base_dataset_path}/models_col1/{region_label}/final_model.keras"
        gcs_metrics_path = f"gs://{base_dataset_path}/models_col1/{region_label}/training_metrics.json"

        try:
            subprocess.run(["gsutil", "cp", model_file, gcs_model_path], check=True)
            log.log_message(f"☁️ Uploaded {model_file} → {gcs_model_path}", stage="training")

            subprocess.run(["gsutil", "cp", metrics_file, gcs_metrics_path], check=True)
            log.log_message(f"☁️ Uploaded {metrics_file} → {gcs_metrics_path}", stage="training")
        except subprocess.CalledProcessError as e:
            log.log_message(f"❌ Error uploading to GCS: {e}", stage="training", level="error")

        log.resources()
        log.log_message(f"🏁 Training session cleared for region {region}", stage="training")

        # --- Cleanup for memory safety ---
        tf.keras.backend.clear_session()
        del model, x_train, y_train, history

    # --- Final summary ---
    total_duration = round((time.time() - start_time_all) / 60, 2)
    log.summary("completed")
    log.log_message(f"✅ All regional training tasks completed in {total_duration} min", stage="training")


# --- Safe direct execution in Jupyter/Colab ---
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        log.log_message("🔹 Module A_2_1 ready for use via GUI", stage="training")
except Exception:
    pass
