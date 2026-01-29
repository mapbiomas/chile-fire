# ==========================================
# A_2_0_simple_gui_train_tensorflow_models.py
# TensorFlow 2.x – GUI for model training
# Fully aligned with bucket naming conventions
# ==========================================

import os
import sys
import subprocess
import numpy as np
import ipywidgets as widgets
from IPython.display import display

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
from A_2_1_training_tensorflow_model_per_region import run_training


def list_gcs_files(bucket_path):
    """List all files in a GCS folder using gsutil."""
    try:
        result = subprocess.run(["gsutil", "ls", bucket_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.splitlines()
        return []
    except Exception as e:
        log.log_message(f"⚠️ Could not list GCS files: {e}", stage="gui", level="warning")
        return []


def extract_model_prefixes(sample_files, country):
    """
    Extracts region/model identifiers based on real training sample file names.
    Example:
    samples_fire_v1_l78_chile_r5_valdivian_temperate_forest_2016.tif
    -> col1_chile_v1_r5_rnn_lstm_ckpt
    """
    model_prefixes = set()
    for f in sample_files:
        base = os.path.basename(f)
        if base.startswith("samples_fire_") and base.endswith(".tif"):
            parts = base.split("_")
            try:
                version = parts[2]  # e.g., v1, v2, v3...
                region = [p for p in parts if p.startswith("r") and p[1:].isdigit()][0]
                prefix = f"col1_{country}_{version}_{region}_rnn_lstm_ckpt"
                model_prefixes.add(prefix)
            except Exception:
                continue
    return sorted(model_prefixes)


def build_interface():
    """Builds the GUI for TensorFlow model training based on available samples."""

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    # Retrieve parameters defined in the notebook
    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", "mapbiomas-fire")

    # --- Detect available model prefixes from training samples ---
    samples_path = f"gs://{base_dataset_path}/training_samples/"
    sample_files = list_gcs_files(samples_path)
    available_models = extract_model_prefixes(sample_files, country)
    available_models = sorted(set(available_models)) 

    if not available_models:
        log.log_message("⚠️ No region samples detected — training GUI will not start.", stage="gui", level="warning")
        print("⚠️ No available regions detected in GCS training_samples folder.")
        return

    # --- Detect existing models in GCS ---
    models_path = f"gs://{base_dataset_path}/models_col1/"
    model_files = list_gcs_files(models_path)

    trained_v2 = []  # TFv2 (.keras/.h5)
    trained_v1 = []  # TFv1 (.ckpt/.meta/.index)

    for model_prefix in available_models:
        v2_match = any(
            f"{model_prefix}" in f and (f.endswith(".h5") or f.endswith(".keras"))
            for f in model_files
        )
        v1_match = any(
            f"{model_prefix}" in f and (
                f.endswith(".ckpt.index") or f.endswith(".ckpt.meta") or f.endswith(".ckpt.data-00000-of-00001")
            )
            for f in model_files
        )
        if v2_match:
            trained_v2.append(model_prefix)
        elif v1_match:
            trained_v1.append(model_prefix)

    # --- Build GUI checkboxes ---
    region_checkboxes = []
    for model_prefix in available_models:
        if model_prefix in trained_v2:
            label = f"⚠️ {model_prefix}"
        elif model_prefix in trained_v1:
            label = f"⚡ {model_prefix}"
        else:
            label = model_prefix
        region_checkboxes.append(widgets.Checkbox(value=False, description=label))

    version_text = widgets.Text(value="v1", description="Version:")
    train_button = widgets.Button(description="Train Models", button_style="success", icon="rocket")
    output_area = widgets.Output()

    # Compact legend
    info_text = widgets.HTML(
        "<p style='font-size:13px;'>"
        "<span style='color:#b58900;'>⚠️</span> Overwrites existing model. "
        "<span style='color:#cb4b16; margin-left:15px;'>⚡</span> Legacy TFv1 — retraining creates new model."
        "</p>"
    )

    # ----------------------------------
    def train_models_click(b):
        with output_area:
            output_area.clear_output()

            version = version_text.value.strip()
            selected_models = [
                cb.description.replace("⚠️ ", "").replace("⚡ ", "")
                for cb in region_checkboxes
                if cb.value
            ]

            if not selected_models:
                print("⚠️ Please select at least one model to train.")
                return

            log.log_message("🚀 Training manually triggered via GUI", stage="gui")
            log.log_message(f"Country: {country}", stage="gui")
            log.log_message(f"Base dataset path: {base_dataset_path}", stage="gui")
            log.log_message(f"Version: {version}", stage="gui")
            log.log_message(f"Selected models: {selected_models}", stage="gui")

            overwrite_models = [m for m in selected_models if m in trained_v2]
            legacy_models = [m for m in selected_models if m in trained_v1]

            if overwrite_models:
                log.log_message(f"⚠️ Overwriting TFv2 models: {overwrite_models}", stage="gui")
            if legacy_models:
                log.log_message(f"⚡ Retraining TFv1 models: {legacy_models}", stage="gui")

            # Placeholder dummy data
            regions_data = {}
            for model_name in selected_models:
                x_train = np.random.rand(500, 10)
                y_train = np.random.randint(0, 2, 500)
                regions_data[model_name] = (x_train, y_train)

            try:
                run_training(
                    regions_data=regions_data,
                    country=country,
                    bucket_name=bucket_name,
                    base_dataset_path=base_dataset_path,
                    overwrite=True
                )
                log.log_message("✅ Model training completed successfully", stage="gui")
            except Exception as e:
                log.log_message(f"❌ Error during training: {e}", stage="gui", level="error")
                print("An error occurred:", e)

    train_button.on_click(train_models_click)

    # Layout
    region_box = widgets.VBox(region_checkboxes)
    display(
        widgets.VBox([
            version_text,
            widgets.HTML("<b>Select models for training:</b>"),
            region_box,
            info_text,
            train_button,
            output_area
        ])
    )


def launch_training_gui():
    """Initialize GUI."""
    log.log_message("🧩 Building training GUI (A_2_0)", stage="gui")
    build_interface()
    log.log_message("✅ GUI ready – waiting for user to click 'Train Models'", stage="gui")


# --- Auto-detect Jupyter/Colab ---
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        launch_training_gui()
except Exception:
    pass
