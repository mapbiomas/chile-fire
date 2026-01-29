# ==========================================
# A_3_1_tensorflow_classification_burned_area.py
# TensorFlow 2.x – Burned Area Classification
# Compatible with vector-based (RNN/MLP) models
# ==========================================

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import rasterio
from rasterio.transform import from_origin
import subprocess
from datetime import datetime
import logging

# Silence absl/tf warnings
logging.getLogger("absl").setLevel(logging.ERROR)

# --- ensure eager execution for tf.data pipelines ---
tf.data.experimental.enable_debug_mode()

# --- Safe import path handling ---
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

# --- Local imports ---
from A_0_2_log_algorithm_monitor import log


# ==========================================================
# Utility functions
# ==========================================================
def list_gcs_files(bucket_path):
    """List all files in a GCS folder using gsutil."""
    try:
        result = subprocess.run(["gsutil", "ls", bucket_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.splitlines()
        return []
    except Exception as e:
        log.log_message(f"⚠️ Could not list GCS files: {e}", stage="classify", level="warning")
        return []


def load_mosaic_from_gcs(gcs_path, local_tmp="temp_mosaic.tif"):
    """Download mosaic from GCS and return local path."""
    try:
        subprocess.run(["gsutil", "cp", gcs_path, local_tmp], check=True)
        return local_tmp
    except subprocess.CalledProcessError as e:
        log.log_message(f"❌ Failed to download mosaic {gcs_path}: {e}", stage="classify", level="error")
        return None


def upload_to_gcs(local_path, gcs_path):
    """Upload a file to GCS."""
    try:
        subprocess.run(["gsutil", "-m", "cp", local_path, gcs_path], check=True)
        log.log_message(f"📤 Uploaded classified raster to {gcs_path}", stage="classify")
    except subprocess.CalledProcessError as e:
        log.log_message(f"❌ Failed to upload result: {e}", stage="classify", level="error")


# ==========================================================
# Classification logic
# ==========================================================
def execute_burned_area_classification():
    """Executes burned area classification using selected models and mosaics."""
    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", "mapbiomas-fire")
    selected_models = global_vars.get("SELECTED_MODELS", [])
    selected_mosaics = global_vars.get("SELECTED_MOSAICS", [])
    version = global_vars.get("CLASSIFICATION_VERSION", "v1")

    if not selected_models or not selected_mosaics:
        print("⚠️ No models or mosaics selected. Please select them in the interface first.")
        return

    log.log_message("🚀 Starting burned area classification", stage="classify")
    log.log_message(f"Country: {country}", stage="classify")
    log.log_message(f"Version: {version}", stage="classify")
    log.log_message(f"Models: {selected_models}", stage="classify")
    log.log_message(f"Mosaics: {selected_mosaics}", stage="classify")

    models_path = f"gs://{base_dataset_path}/models_col1/"
    mosaics_path = f"gs://{base_dataset_path}/mosaics_col1_cog/"
    output_path = f"gs://{base_dataset_path}/result_classified/"

    # ======================================================
    for model_name in selected_models:
        model_file = f"{models_path}{model_name}.h5"
        local_model = f"{model_name}.h5"

        try:
            subprocess.run(["gsutil", "cp", model_file, local_model], check=True)
            model = tf.keras.models.load_model(local_model, compile=False)
            log.log_message(f"🧠 Loaded TF2 model: {model_name}", stage="classify")
        except Exception as e:
            log.log_message(f"❌ Could not load model {model_name}: {e}", stage="classify", level="error")
            continue

        # Determine region ID (e.g., r2) to match mosaics
        region_id = None
        parts = model_name.split("_")
        for p in parts:
            if p.startswith("r") and p[1:].isdigit():
                region_id = p
                break

        mosaics_for_model = [m for m in selected_mosaics if region_id and region_id in m]

        for mosaic_name in mosaics_for_model:
            log.log_message(f"🧩 Classifying {mosaic_name} with model {model_name}", stage="classify")
            gcs_mosaic = f"{mosaics_path}{mosaic_name}.tif"
            local_mosaic = load_mosaic_from_gcs(gcs_mosaic, f"{mosaic_name}.tif")
            if not local_mosaic:
                continue

            try:
                with rasterio.open(local_mosaic) as src:
                    img = src.read().astype(np.float32)
                    transform = src.transform
                    profile = src.profile

                # (bands, height, width) -> (height*width, bands)
                img = np.transpose(img, (1, 2, 0))
                n_rows, n_cols, n_bands = img.shape
                flat_pixels = img.reshape(-1, n_bands)

                # Normalize
                flat_pixels = np.nan_to_num(flat_pixels)
                flat_pixels = (flat_pixels - np.min(flat_pixels, axis=0)) / (
                    np.max(flat_pixels, axis=0) - np.min(flat_pixels, axis=0) + 1e-6
                )

                # Predict in batches
                batch_size = 4096
                preds = []
                for i in range(0, len(flat_pixels), batch_size):
                    batch = flat_pixels[i:i + batch_size]
                    y_pred = model.predict(batch, verbose=0)
                    if y_pred.ndim > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    preds.extend(y_pred)

                preds = np.array(preds).reshape(n_rows, n_cols).astype(np.uint8)

                # Save output raster
                out_file = f"classified_{mosaic_name}.tif"
                profile.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(out_file, "w", **profile) as dst:
                    dst.write(preds, 1)

                # Upload result
                gcs_out = f"{output_path}{out_file}"
                upload_to_gcs(out_file, gcs_out)

                log.log_message(f"✅ Classified {mosaic_name} successfully", stage="classify")

            except Exception as e:
                log.log_message(f"❌ Error in model {model_name}: {e}", stage="classify", level="error")

    log.log_message("✅ Burned area classification completed", stage="classify")
    log.summary("completed")
    print("✅ Classification completed successfully.")
