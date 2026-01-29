# ==========================================
# A_2_0_simple_gui_train_tensorflow_models.py
# TensorFlow 2.x – Training GUI with model detection (TFv1 vs TFv2)
# ==========================================

import os
import glob
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from A_0_2_log_algorithm_monitor import log
from A_2_1_training_tensorflow_model_per_region import run_training


def build_interface():
    """
    Creates the interactive GUI for selecting and training models.
    - Detects existing models (both legacy TFv1 and new TFv2 formats)
    - Warns about potential overwrites
    - Keeps full compatibility with the MapBiomas Fire notebook structure
    """

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    # Retrieve global parameters (set earlier in A_0_1)
    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", None)

    # Example regions — replace with dynamic list if available
    available_regions = ["r01", "r02", "r03", "r04", "r05"]

    # Detect trained models (both formats)
    trained_regions_v2 = []  # New TFv2 models (.h5)
    trained_regions_v1 = []  # Legacy TFv1 models (.ckpt)
    model_dir = os.path.join(base_dataset_path, "models_col1")

    for region in available_regions:
        # Patterns for legacy TFv1 and current TFv2
        tfv1_patterns = glob.glob(os.path.join(model_dir, f"*_{country}_*_{region}_rnn_lstm_ckpt.*"))
        tfv2_patterns = glob.glob(os.path.join(model_dir, f"*{region}*model_final.h5"))

        if tfv2_patterns:
            trained_regions_v2.append(region)
        elif tfv1_patterns:
            trained_regions_v1.append(region)

    # Build region checkboxes with emoji labels
    region_checkboxes = []
    for region in available_regions:
        if region in trained_regions_v2:
            label = f"⚠️ {region}"  # Trained (compatible)
        elif region in trained_regions_v1:
            label = f"⚡ {region}"  # Legacy TFv1 model
        else:
            label = region
        region_checkboxes.append(widgets.Checkbox(value=False, description=label))

    # Input field for model version (preserved)
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

    # Output area for logs
    output_area = widgets.Output()

    # Informative legend for users
    overwrite_info = widgets.HTML(
        "<p style='font-size:13px;'>"
        "<span style='color:#b58900;'>⚠️</span> Regions marked with this symbol already have a <b>compatible trained model</b>. "
        "Running training on them will <b>overwrite</b> the existing models.<br>"
        "<span style='color:#cb4b16;'>⚡</span> Regions marked with this symbol contain <b>legacy TensorFlow v1 models</b> "
        "that are <b>not compatible</b> with the current training format. Retraining will create a new version."
        "</p>"
    )

    # ----------------------------------
    # Callback executed when "Train Models" is clicked
    # ----------------------------------
    def train_models_click(b):
        with output_area:
            output_area.clear_output()

            version = version_text.value.strip()
            selected_regions = [cb.description.replace("⚠️ ", "").replace("⚡ ", "") for cb in region_checkboxes if cb.value]

            if not selected_regions:
                print("⚠️ Please select at least one region to train.")
                return

            log.log_message("🚀 Training manually triggered via GUI", stage="gui")
            log.log_message(f"Country: {country}", stage="gui")
            log.log_message(f"Dataset base path: {base_dataset_path}", stage="gui")
            log.log_message(f"Version: {version}", stage="gui")
            log.log_message(f"Regions selected: {selected_regions}", stage="gui")

            # Detect special statuses
            overwrite_regions = [r for r in selected_regions if r in trained_regions_v2]
            legacy_regions = [r for r in selected_regions if r in trained_regions_v1]

            if overwrite_regions:
                log.log_message(f"⚠️ These regions already have compatible models and will be overwritten: {overwrite_regions}", stage="gui")
            if legacy_regions:
                log.log_message(f"⚡ These regions have legacy TFv1 models and will be retrained from scratch: {legacy_regions}", stage="gui")

            # Simulated training data (replace with actual data loading)
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

    # Bind callback
    train_button.on_click(train_models_click)

    # Compose the GUI layout
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
    """Initialize the GUI (entry point for A_2_0)."""
    log.log_message("🧩 Building training GUI (A_2_0)", stage="gui")
    build_interface()
    log.log_message("✅ GUI ready – waiting for user to click 'Train Models'", stage="gui")


# ----------------------------------
# Auto-detect environment (Colab/Jupyter)
# ----------------------------------
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        launch_training_gui()
except Exception:
    pass
