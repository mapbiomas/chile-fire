# ==========================================
# A_3_0_simple_gui_train_tensorflow_classification.py
# TensorFlow 2.x – GUI for burned area classification (no button)
# Selections stored in global variables for A_3_1 execution
# ==========================================

import os
import sys
import subprocess
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


# ==========================================================
# Helper functions
# ==========================================================
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


def extract_model_names(model_files, country):
    """Extracts model names (col1_{country}_vX_rY_rnn_lstm_ckpt.h5)"""
    models = set()
    for f in model_files:
        base = os.path.basename(f)
        if base.endswith(".h5") or base.endswith(".keras"):
            if base.startswith(f"col1_{country}_"):
                models.add(base.replace(".h5", "").replace(".keras", ""))
    return sorted(models)


def extract_mosaic_names(mosaic_files):
    """Extracts mosaic file names (e.g., mosaic_col1_v1_r1_2016_cog.tif)"""
    mosaics = set()
    for f in mosaic_files:
        base = os.path.basename(f)
        if base.endswith(".tif"):
            mosaics.add(base.replace(".tif", ""))
    return sorted(mosaics)


# ==========================================================
# GUI builder
# ==========================================================
def build_classification_gui():
    """Builds GUI for selecting models and mosaics for classification."""

    from IPython import get_ipython
    global_vars = get_ipython().user_ns

    country = global_vars.get("country", "unknown")
    base_dataset_path = global_vars.get("BASE_DATASET_PATH", "")
    bucket_name = global_vars.get("bucket_name", "mapbiomas-fire")

    log.log_message("🧩 Building classification GUI (A_3_0)", stage="gui")

    # --- Load models and mosaics from GCS ---
    models_path = f"gs://{base_dataset_path}/models_col1/"
    mosaics_path = f"gs://{base_dataset_path}/mosaics_col1_cog/"
    results_path = f"gs://{base_dataset_path}/result_classified/"

    model_files = list_gcs_files(models_path)
    mosaic_files = list_gcs_files(mosaics_path)
    result_files = list_gcs_files(results_path)

    available_models = extract_model_names(model_files, country)
    available_mosaics = extract_mosaic_names(mosaic_files)
    classified_mosaics = extract_mosaic_names(result_files)

    if not available_models:
        log.log_message("⚠️ No trained models detected — classification GUI will not start.", stage="gui", level="warning")
        print("⚠️ No trained models detected in GCS models_col1 folder.")
        return

    if not available_mosaics:
        log.log_message("⚠️ No mosaics detected — classification GUI will not start.", stage="gui", level="warning")
        print("⚠️ No mosaics detected in GCS mosaics_col1_cog folder.")
        return

    # --- Build checkboxes for models ---
    model_checkboxes = [
        widgets.Checkbox(value=False, description=model_name)
        for model_name in available_models
    ]

    # --- Build checkboxes for mosaics ---
    mosaic_checkboxes = []
    for mosaic_name in available_mosaics:
        if mosaic_name in classified_mosaics:
            label = f"⚠️ {mosaic_name}"
        else:
            label = f"✳️ {mosaic_name}"
        mosaic_checkboxes.append(widgets.Checkbox(value=False, description=label))

    version_text = widgets.Text(value="v1", description="Version:")

    # Compact legend
    info_text = widgets.HTML(
        "<p style='font-size:13px;'>"
        "<span style='color:#b58900;'>⚠️</span> Already classified"
        "<span style='margin-left:20px; color:#268bd2;'>✳️</span> New mosaics"
        "</p>"
    )

    # ======================================================
    # Live update globals based on selections
    # ======================================================
    def update_globals(change=None):
        selected_models = [cb.description for cb in model_checkboxes if cb.value]
        selected_mosaics = [
            cb.description.replace("⚠️ ", "").replace("✳️ ", "")
            for cb in mosaic_checkboxes if cb.value
        ]
        version = version_text.value.strip()

        global_vars["SELECTED_MODELS"] = selected_models
        global_vars["SELECTED_MOSAICS"] = selected_mosaics
        global_vars["CLASSIFICATION_VERSION"] = version

    for cb in model_checkboxes + mosaic_checkboxes:
        cb.observe(update_globals, "value")

    version_text.observe(update_globals, "value")

    # Initial assignment
    update_globals()

    # ======================================================
    # Display interface
    # ======================================================
    display(
        widgets.VBox([
            version_text,
            widgets.HTML("<b>Select trained models:</b>"),
            widgets.VBox(model_checkboxes),
            widgets.HTML("<b>Select mosaics for classification:</b>"),
            widgets.VBox(mosaic_checkboxes),
            info_text,
            widgets.HTML(
                "<p style='font-size:13px; color:gray;'>"
                "Selections are stored globally. "
                "After selecting, run <code>execute_burned_area_classification()</code> to start."
                "</p>"
            )
        ])
    )

    log.log_message("✅ GUI ready – waiting for user to run execute_burned_area_classification()", stage="gui")


# ==========================================================
# Entry point
# ==========================================================
def launch_classification_gui():
    """Initialize classification GUI."""
    build_classification_gui()


try:
    from IPython import get_ipython
    if get_ipython() is not None:
        launch_classification_gui()
except Exception:
    pass
