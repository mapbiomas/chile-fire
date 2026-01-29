# ==========================================
# A_3_1_tensorflow_classification_burned_area.py
# TensorFlow 2.x – burned area classification and GCS export
# Works with GUI selections defined in A_3_0
# ==========================================

import os
import sys
import time
import subprocess
import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf
from datetime import datetime
from scipy.ndimage import binary_dilation, binary_erosion

# Enable eager mode for tf.data pipelines
tf.data.experimental.enable_debug_mode()

# --- Safe import path handling for Colab runtime resets ---
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
def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def upload_to_gcs(local_path, gcs_path):
    """Upload file to Google Cloud Storage."""
    try:
        subprocess.run(["gsutil", "-m", "cp", local_path, gcs_path], check=True)
        log.log_message(f"📤 Uploaded: {gcs_path}", stage="classification")
    except subprocess.CalledProcessError as e:
        log.log_message(f"⚠️ Upload failed: {e}", stage="classification", level="error")


def load_model_compatible(model_name, base_dataset_path):
    """Load either TF2 (.h5) or legacy TF1 (.ckpt) model."""
    gcs_model_path = f"gs://{base_dataset_path}/models_col1/{model_name}.h5"
    local_model_path = f"./models_col1/{model_name}.h5"

    # Try downloading model file
    ensure_dir("./models_col1")
    subprocess.run(["gsutil", "cp", gcs_model_path, local_model_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if os.path.exists(local_model_path):
        model = tf.keras.models.load_model(local_model_path)
        log.log_message(f"🧠 Loaded TF2 model: {model_name}", stage="classification")
        return model

    # Try TFv1 legacy checkpoint
    ckpt_prefix = f"gs://{base_dataset_path}/models_col1/{model_name}_ckpt"
    local_ckpt_prefix = f"./models_col1/{model_name}_ckpt"

    subprocess.run(["gsutil", "-m", "cp", f"{ckpt_prefix}.*", "./models_col1/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(f"{local_ckpt_prefix}.index"):
        log.log_message(f"⚡ Loaded legacy TFv1 model: {model_name}", stage="classification")
        model = tf.compat.v1.keras.models.load_model(local_ckpt_prefix)
        return model

    raise FileNotFoundError(f"Model not found: {model_name}")


def classify_mosaic(model, mosaic_path, output_path, smooth_kernel=3, simulate_test=False):
    """Perform inference on mosaic and save classified TIFF."""

    with rasterio.open(mosaic_path) as src:
        profile = src.profile.copy()
        if simulate_test:
            width, height = min(256, src.width), min(256, src.height)
            window = Window(0, 0, width, height)
            image = src.read(window=window).astype(np.float32)
            log.log_message(f"🔍 Simulation mode active (subset {width}x{height})", stage="classification")
        else:
            image = src.read().astype(np.float32)

    # Normalize and reshape
    image = np.transpose(image, (1, 2, 0))  # (H, W, bands)
    image = np.nan_to_num(image)
    image = (image - np.mean(image)) / (np.std(image) + 1e-6)

    # Predict in blocks
    block_size = 256
    height, width, _ = image.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            input_data = np.expand_dims(block, axis=0)
            preds = model.predict(input_data, verbose=0)
            preds = np.argmax(preds, axis=-1)[0]
            output[y:y+preds.shape[0], x:x+preds.shape[1]] = preds.astype(np.uint8)

    # Postprocess
    output = binary_dilation(output, iterations=smooth_kernel)
    output = binary_erosion(output, iterations=smooth_kernel)

    # Save GeoTIFF
    ensure_dir(os.path.dirname(output_path))
    profile.update(dtype=rasterio.uint8, count=1, compress="LZW")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)

    log.log_message(f"💾 Saved classified mosaic: {output_path}", stage="classification")


# ==========================================================
# Main execution function
# ==========================================================
def execute_burned_area_classification(simulate_test=False):
    """Main classification execution function (called manually)."""

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    selected_models = global_vars.get("SELECTED_MODELS", [])
    selected_mosaics = global_vars.get("SELECTED_MOSAICS", [])
    version = global_vars.get("CLASSIFICATION_VERSION", "v1")
    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", "mapbiomas-fire")

    if not selected_models or not selected_mosaics:
        print("⚠️ No models or mosaics selected. Please run A_3_0 first.")
        return

    log.log_message("🚀 Starting burned area classification", stage="classification")
    log.log_message(f"Country: {country}", stage="classification")
    log.log_message(f"Version: {version}", stage="classification")
    log.log_message(f"Models: {selected_models}", stage="classification")
    log.log_message(f"Mosaics: {selected_mosaics}", stage="classification")

    for model_name in selected_models:
        try:
            model = load_model_compatible(model_name, base_dataset_path)

            for mosaic_name in selected_mosaics:
                log.log_message(f"🧩 Classifying {mosaic_name} with model {model_name}", stage="classification")

                gcs_mosaic_path = f"gs://{base_dataset_path}/mosaics_col1_cog/{mosaic_name}.tif"
                local_mosaic_path = f"./mosaics/{mosaic_name}.tif"
                ensure_dir("./mosaics")

                subprocess.run(["gsutil", "cp", gcs_mosaic_path, local_mosaic_path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                output_local = f"./classified/{model_name}_{mosaic_name}_classified.tif"
                classify_mosaic(model, local_mosaic_path, output_local, simulate_test=simulate_test)

                gcs_output_path = f"gs://{base_dataset_path}/result_classified/{model_name}_{mosaic_name}_classified.tif"
                upload_to_gcs(output_local, gcs_output_path)

        except Exception as e:
            log.log_message(f"❌ Error in model {model_name}: {e}", stage="classification", level="error")

    log.log_message("✅ Burned area classification completed", stage="classification")
    print("✅ Classification completed successfully.")


# ==========================================================
# Optional test trigger
# ==========================================================
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        log.log_message("🔹 Module A_3_1 ready — use execute_burned_area_classification()", stage="classification")
except Exception:
    pass
