# ==========================================
# A_2_0_simple_gui_train_tensorflow_models.py
# TensorFlow 2.x – Dynamic GUI for training, consistent with original pipeline
# ==========================================

# --- Handle imports safely for Colab runtime resets ---
import os
import sys
import subprocess
import numpy as np
import ipywidgets as widgets
from IPython.display import display

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

# --- Local imports (now work even after reset) ---
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


def build_interface():
    """Builds the GUI for TensorFlow model training based on available sample regions."""

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    # Retrieve parameters defined in the notebook
    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", "mapbiomas-fire")

    # --- Detect available regions from samples ---
    samples_path = f"gs://{base_dataset_path}/samples/"
    sample_files = list_gcs_files(samples_path)

    available_regions = sorted(set([
        f.split("_r")[-1][:2]  # detect r01, r02, etc.
        for f in sample_files
        if "_r" in f
    ]))
    available_regions = [f"r{r}" if not r.startswith("r") else r for r in available_regions]

    if not available_regions:
        log.log_message("⚠️ No region samples detected — training GUI will not start.", stage="gui", level="warning")
        print("⚠️ No available regions detected in GCS samples folder.")
        return

    # --- Detect existing models from models_col1 folder ---
    models_path = f"gs://{base_dataset_path}/models_col1/"
    model_files = list_gcs_files(models_path)

    trained_regions_v2 = []  # TFv2 (.h5)
    trained_regions_v1 = []  # TFv1 (.ckpt, .meta, .index)

    for region in available_regions:
        region_short = region.lstrip("r0")

        tfv1_match = any(
            f"col1_{country}_" in f and (
                f"_r{region_short}_" in f or f"_{region}_"
            ) and "rnn_lstm_ckpt" in f
            for f in model_files
        )

        tfv2_match = any(
            f"col1_{country}_" in f and (
                f"_r{region_short}_" in f or f"_{region}_"
            ) and f.endswith("model_final.h5")
            for f in model_files
        )

        if tfv2_match:
            trained_regions_v2.append(region)
        elif tfv1_match:
            trained_regions_v1.append(region)

    # --- Build the GUI ---
    region_checkboxes = []
    for region in available_regions:
        if region in trained_regions_v2:
            label = f"⚠️ {region}"
        elif region in trained_regions_v1:
            label = f"⚡ {region}"
        else:
            label = region
        region_checkboxes.append(widgets.Checkbox(value=False, description=label))

    version_text = widgets.Text(value="v1", description="Version:")
    train_button = widgets.Button(description="Train Models", button_style="success", icon="rocket")
    output_area = widgets.Output()

    # compact legend
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
            selected_regions = [
                cb.description.replace("⚠️ ", "").replace("⚡ ", "")
                for cb in region_checkboxes
                if cb.value
            ]

            if not selected_regions:
                print("⚠️ Please select at least one region to train.")
                return

            log.log_message("🚀 Training manually triggered via GUI", stage="gui")
            log.log_message(f"Country: {country}", stage="gui")
            log.log_message(f"Base dataset path: {base_dataset_path}", stage="gui")
            log.log_message(f"Version: {version}", stage="gui")
            log.log_message(f"Selected regions: {selected_regions}", stage="gui")

            overwrite_regions = [r for r in selected_regions if r in trained_regions_v2]
            legacy_regions = [r for r in selected_regions if r in trained_regions_v1]

            if overwrite_regions:
                log.log_message(f"⚠️ Overwriting TFv2 models: {overwrite_regions}", stage="gui")
            if legacy_regions:
                log.log_message(f"⚡ Retraining TFv1 models: {legacy_regions}", stage="gui")

            # Placeholder data (your real loader replaces this)
            regions_data = {}
            for region in selected_regions:
                x_train = np.random.rand(500, 10)
                y_train = np.random.randint(0, 2, 500)
                regions_data[region] = (x_train, y_train)

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
            widgets.HTML("<b>Select regions for training:</b>"),
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
