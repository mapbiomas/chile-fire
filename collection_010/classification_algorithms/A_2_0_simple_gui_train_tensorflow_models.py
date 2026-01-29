# ==========================================
# A_2_0_simple_gui_train_tensorflow_models.py
# TensorFlow 2.x – GUI with overwrite warning per region
# ==========================================

import os
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from A_0_2_log_algorithm_monitor import log
from A_2_1_training_tensorflow_model_per_region import run_training


def build_interface():
    """Creates the GUI for selecting regions and training models with overwrite warnings."""

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    # Retrieve variables defined in A_0_1
    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", None)

    # Example regions — can be replaced by your dynamic list
    available_regions = ["r01", "r02", "r03", "r04"]

    # Detect which regions already have trained models
    trained_regions = []
    for region in available_regions:
        model_path = os.path.join(base_dataset_path, "models_col1", "model_final.h5")
        if os.path.exists(model_path):
            trained_regions.append(region)

    # Build region checkboxes
    region_checkboxes = []
    for region in available_regions:
        label = f"⚠️ {region}" if region in trained_regions else region
        region_checkboxes.append(widgets.Checkbox(value=False, description=label))

    # Input for version (kept for completeness)
    version_text = widgets.Text(
        value="v1",
        description="Version:"
    )

    # Train button
    train_button = widgets.Button(
        description="Train Models",
        button_style="success",
        icon="rocket"
    )

    # Output area
    output_area = widgets.Output()

    # Information banner
    overwrite_info = widgets.HTML(
        "<p style='color: #b58900; font-size: 13px;'>⚠️ Regions marked with a warning symbol already have a trained model.<br>"
        "Running training on them will overwrite the existing models.</p>"
    )

    # -----------------------------
    # Callback when clicking Train
    # -----------------------------
    def train_models_click(b):
        with output_area:
            output_area.clear_output()

            version = version_text.value
            regions = [cb.description.replace("⚠️ ", "") for cb in region_checkboxes if cb.value]

            if not regions:
                print("⚠️ Please select at least one region to train.")
                return

            overwrite_selected = [r for r in regions if r in trained_regions]
            log.log_message("🚀 Training manually triggered via GUI", stage="gui")
            log.log_message(f"Country: {country}", stage="gui")
            log.log_message(f"Dataset base path: {base_dataset_path}", stage="gui")
            log.log_message(f"Version: {version}", stage="gui")
            log.log_message(f"Regions selected: {regions}", stage="gui")

            if overwrite_selected:
                log.log_message(f"⚠️ These models will be overwritten: {overwrite_selected}", stage="gui")

            # Simulated data (replace with real samples if needed)
            regions_data = {}
            for region in regions:
                x_train = np.random.rand(500, 10)
                y_train = np.random.randint(0, 2, 500)
                regions_data[region] = (x_train, y_train)

            try:
                run_training(
                    regions_data=regions_data,
                    country=country,
                    bucket_name=bucket_name,
                    base_dataset_path=base_dataset_path,
                    overwrite=True  # always overwrite if region selected
                )
                log.log_message("✅ Model training completed successfully", stage="gui")
            except Exception as e:
                log.log_message(f"❌ Error during GUI training: {e}", stage="gui", level="error")
                print("An error occurred:", e)

    # Bind callback
    train_button.on_click(train_models_click)

    # Layout
    region_box = widgets.VBox(region_checkboxes)
    display(
        widgets.VBox([
            version_text,
            widgets.HTML("<b>Select regions for training:</b>"),
            region_box,
            overwrite_info,
            train_button,
            output_area
        ])
    )


def launch_training_gui():
    """Initializes the GUI for model training."""
    log.log_message("🧩 Building training GUI (A_2_0)", stage="gui")
    build_interface()
    log.log_message("✅ GUI ready – waiting for user to click 'Train Models'", stage="gui")


# -----------------------------
# Auto-detect Jupyter/Colab
# -----------------------------
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        launch_training_gui()
except Exception:
    pass
