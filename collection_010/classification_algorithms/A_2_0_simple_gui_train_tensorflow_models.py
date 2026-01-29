# ==========================================
# A_2_0_simple_gui_train_tensorflow_models.py
# TensorFlow 2.x – GUI with region checkboxes and overwrite option
# ==========================================

import os
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from A_0_2_log_algorithm_monitor import log
from A_2_1_training_tensorflow_model_per_region import run_training


# -----------------------------
# GUI builder
# -----------------------------
def build_interface():
    """Creates the GUI for selecting regions and training models."""

    # Simulate available regions (replace with your list of valid regions)
    available_regions = ["r01", "r02", "r03", "r04"]

    # Build region checkboxes
    region_checkboxes = [
        widgets.Checkbox(value=False, description=region) for region in available_regions
    ]

    # Overwrite warning checkbox
    overwrite_checkbox = widgets.Checkbox(
        value=False,
        description="⚠️ Overwrite existing models",
        indent=False
    )

    # Version input
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

    # -----------------------------
    # Button callback
    # -----------------------------
    def train_models_click(b):
        with output_area:
            output_area.clear_output()
            from IPython import get_ipython
            global_vars = get_ipython().user_ns

            # Retrieve global variables defined in notebook
            country = global_vars.get("country", "unknown")
            bucket_name = global_vars.get("bucket_name", None)
            base_subfolder = global_vars.get("BASE_SUBFOLDER", "")
            base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")

            version = version_text.value
            overwrite = overwrite_checkbox.value

            # Collect selected regions
            regions = [cb.description for cb in region_checkboxes if cb.value]

            if not regions:
                print("⚠️ Please select at least one region to train.")
                return

            log.log_message("🚀 Training manually triggered via GUI", stage="gui")
            log.log_message(f"Country: {country}", stage="gui")
            log.log_message(f"Dataset base path: {base_dataset_path}", stage="gui")
            log.log_message(f"Version: {version}", stage="gui")
            log.log_message(f"Regions selected: {regions}", stage="gui")
            log.log_message(f"Overwrite enabled: {overwrite}", stage="gui")

            # Simulate region data (replace with actual data loading logic)
            regions_data = {}
            for region in regions:
                x_train = np.random.rand(500, 10)
                y_train = np.random.randint(0, 2, 500)
                regions_data[region] = (x_train, y_train)

            try:
                run_training(regions_data, country=country, bucket_name=bucket_name)
                log.log_message("✅ Model training completed successfully", stage="gui")
            except Exception as e:
                log.log_message(f"❌ Error during GUI training: {e}", stage="gui", level="error")
                print("An error occurred:", e)

    train_button.on_click(train_models_click)

    # Display all widgets
    region_box = widgets.VBox(region_checkboxes)
    display(
        widgets.VBox([
            version_text,
            widgets.HTML("<b>Select regions for training:</b>"),
            region_box,
            overwrite_checkbox,
            train_button,
            output_area
        ])
    )


# -----------------------------
# main() – only builds the GUI
# -----------------------------
def main():
    """Initializes the training GUI and waits for user interaction."""
    log.log_message("🧩 Building model training GUI (A_2_0)", stage="gui")
    build_interface()
    log.log_message("✅ GUI ready – waiting for user to click 'Train Models'", stage="gui")
