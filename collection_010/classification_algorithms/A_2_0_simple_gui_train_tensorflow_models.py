# ==========================================
# A_2_0_simple_gui_train_tensorflow_models.py
# TensorFlow 2.x – Training GUI with GCS-based model detection
# ==========================================

import os
import subprocess
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from A_0_2_log_algorithm_monitor import log
from A_2_1_training_tensorflow_model_per_region import run_training


def list_gcs_files(bucket_path):
    """Return list of file paths from a GCS bucket directory."""
    try:
        result = subprocess.run(["gsutil", "ls", bucket_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.splitlines()
        else:
            return []
    except Exception as e:
        log.log_message(f"⚠️ Could not list GCS files: {e}", stage="gui", level="warning")
        return []


def build_interface():
    """Builds the interactive training interface with automatic GCS model detection."""

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    # Retrieve variables defined earlier in A_0_1
    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", None)

    # Example regions (replace with dynamic list if available)
    available_regions = ["r01", "r02", "r03", "r04", "r05"]

    # List files in the GCS models folder
    gcs_models_path = f"gs://{bucket_name}/sudamerica/{country}/models_col1/"
    bucket_files = list_gcs_files(gcs_models_path)

    trained_regions_v2 = []  # TFv2 (.h5)
    trained_regions_v1 = []  # TFv1 (.ckpt)

    for region in available_regions:
        # Detect TFv1 legacy models
        if any(f"_{region}_rnn_lstm_ckpt" in f for f in bucket_files):
            trained_regions_v1.append(region)
        # Detect TFv2 new models
        elif any(f"{region}" in f and "model_final.h5" in f for f in bucket_files):
            trained_regions_v2.append(region)

    # Build region checkboxes
    region_checkboxes = []
    for region in available_regions:
        if region in trained_regions_v2:
            label = f"⚠️ {region}"  # compatible model
        elif region in trained_regions_v1:
            label = f"⚡ {region}"  # legacy TFv1 model
        else:
            label = region
        region_checkboxes.append(widgets.Checkbox(value=False, description=label))

    # Input for version (optional)
    version_text = widgets.Text(value="v1", description="Version:")

    # Train button
    train_button = widgets.Button(
        description="Train Models",
        button_style="success",
        icon="rocket"
    )

    # Output area
    output_area = widgets.Output()

    # Compact information banner
    info_text = widgets.HTML(
        "<p style='font-size:13px;'>"
        "<span style='color:#b58900;'>⚠️</span> Existing compatible model — will overwrite.<br>"
        "<span style='color:#cb4b16;'>⚡</span> Legacy TFv1 model — retraining will replace it."
        "</p>"
    )

    # ----------------------------------
    # Button callback
    # ----------------------------------
    def train_models_click(b):
        with output_area:
            output_area.clear_output()

            version = version_text.value.strip()
            selected_regions = [
                cb.description.replace("⚠️ ", "").replace("⚡ ", "") for cb in region_checkboxes if cb.value
            ]

            if not selected_regions:
                print("⚠️ Please select at least one region to train.")
                return

            log.log_message("🚀 Training manually triggered via GUI", stage="gui")
            log.log_message(f"Country: {country}", stage="gui")
            log.log_message(f"Dataset base path: {base_dataset_path}", stage="gui")
            log.log_message(f"Version: {version}", stage="gui")
            log.log_message(f"Regions selected: {selected_regions}", stage="gui")

            overwrite_regions = [r for r in selected_regions if r in trained_regions_v2]
            legacy_regions = [r for r in selected_regions if r in trained_regions_v1]

            if overwrite_regions:
                log.log_message(f"⚠️ Overwriting existing compatible models: {overwrite_regions}", stage="gui")
            if legacy_regions:
                log.log_message(f"⚡ Retraining legacy TFv1 models: {legacy_regions}", stage="gui")

            # Simulated sample data (replace with real training data)
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

    # Connect button to callback
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
    """Initializes the GUI."""
    log.log_message("🧩 Building training GUI (A_2_0)", stage="gui")
    build_interface()
    log.log_message("✅ GUI ready – waiting for user to click 'Train Models'", stage="gui")


# Auto-detect Jupyter/Colab
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        launch_training_gui()
except Exception:
    pass
