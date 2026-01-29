# ==========================================
# A_0_2_log_algorithm_monitor.py
# LogManager v3 – GCS Upload + Cleanup
# ==========================================

import os
import json
import psutil
import shutil
import datetime
import time
import subprocess
from threading import Lock, Thread
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

class LogManager:
    """
    Centralized logging system for MapBiomas Fire Pipeline.

    Modes:
      - operational: concise logs + periodic resource monitoring
      - debug: more detailed, frequent logs
      - silent: no console output, only JSON file
    """

    def __init__(
        self,
        mode="operational",
        log_dir="./logs",
        log_interval_min=5,
        gcs_bucket=None,
        gcs_path=None
    ):
        self.mode = mode
        self.log_dir = log_dir
        self.log_interval_min = log_interval_min
        self.gcs_bucket = gcs_bucket
        self.gcs_path = gcs_path or ""
        self.log_file = os.path.join(log_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.summary_file = self.log_file.replace(".jsonl", "_summary.json")
        self.lock = Lock()
        self._stop_flag = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        os.makedirs(self.log_dir, exist_ok=True)

        self._print_colored(f"🟢 LogManager initialized in '{mode}' mode. Logs -> {self.log_file}", "info")

        # Background monitor thread
        if self.mode != "silent":
            self.monitor_thread = Thread(target=self._auto_monitor, daemon=True)
            self.monitor_thread.start()

    # ----------------------------------------
    # Public log entry
    # ----------------------------------------
    def log_message(self, message, stage="general", level="info"):
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        data = {"timestamp": timestamp, "stage": stage, "level": level, "message": message}
        self._save_json(data)

        if self.mode != "silent":
            prefix = "🔹" if level == "info" else "⚠️" if level == "warn" else "❌"
            self._print_colored(f"[{timestamp}] {prefix} {message}", level)

    # ----------------------------------------
    # Log resource usage
    # ----------------------------------------
    def resources(self, auto=False):
        vm = psutil.virtual_memory()
        du = shutil.disk_usage('/')
        ram_used = round(vm.used / 1024**3, 2)
        ram_total = round(vm.total / 1024**3, 2)
        disk_used = round(du.used / 1024**3, 2)
        disk_total = round(du.total / 1024**3, 2)
        cpu_percent = psutil.cpu_percent(interval=0.5)
        gpu_info = self._get_gpu_info()

        msg = f"💾 RAM: {ram_used}/{ram_total} GB | Disk: {disk_used}/{disk_total} GB | CPU: {cpu_percent}%"
        if gpu_info:
            msg += f" | GPU: {gpu_info}"

        self.log_message(msg, stage="resources")

        if auto:
            # Silent background save
            data = {
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": "resources",
                "ram_used_gb": ram_used,
                "ram_total_gb": ram_total,
                "disk_used_gb": disk_used,
                "disk_total_gb": disk_total,
                "cpu_percent": cpu_percent,
                "gpu_info": gpu_info,
            }
            self._save_json(data)

    # ----------------------------------------
    # Background monitor
    # ----------------------------------------
    def _auto_monitor(self):
        while not self._stop_flag:
            time.sleep(self.log_interval_min * 60)
            if not self._stop_flag and self.mode in ["operational", "debug"]:
                self.resources(auto=True)

    # ----------------------------------------
    # GPU Info via nvidia-smi
    # ----------------------------------------
    def _get_gpu_info(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 4:
                    name, mem_used, mem_total, util = parts
                    return f"{name.strip()} {mem_used}/{mem_total} MB ({util.strip()}%)"
        except Exception:
            pass
        return None

    # ----------------------------------------
    # Colorized console output
    # ----------------------------------------
    def _print_colored(self, msg, level="info"):
        colors = {
            "info": "\033[94m",
            "warn": "\033[93m",
            "error": "\033[91m",
            "resources": "\033[92m"
        }
        endc = "\033[0m"
        print(f"{colors.get(level, '')}{msg}{endc}")

    # ----------------------------------------
    # Save JSON line
    # ----------------------------------------
    def _save_json(self, data):
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")

    # ----------------------------------------
    # Stop monitor
    # ----------------------------------------
    def stop(self):
        self._stop_flag = True
        self._print_colored("🛑 LogManager background monitor stopped.", "warn")

    # ----------------------------------------
    # Switch mode dynamically
    # ----------------------------------------
    def set_mode(self, mode):
        valid_modes = ["operational", "debug", "silent"]
        if mode not in valid_modes:
            self.log_message(f"Invalid mode '{mode}', keeping '{self.mode}'", level="warn")
            return
        self.mode = mode
        self.log_message(f"Switched logging mode to '{mode}'", level="info")

    # ----------------------------------------
    # Summary + GCS upload
    # ----------------------------------------
    def summary(self, status="completed"):
        end_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        data = {"timestamp": end_time, "status": status}
        with open(self.summary_file, "w") as f:
            json.dump(data, f, indent=2)
        self._save_json(data)
        self._print_colored(f"🧾 Log summary recorded — status: {status}", "info")
        self.stop()

        if self.gcs_bucket:
            self._executor.submit(self._upload_to_gcs, self.log_file)
            self._executor.submit(self._upload_to_gcs, self.summary_file)

    # ----------------------------------------
    # Upload to GCS (with retry)
    # ----------------------------------------
    def _upload_to_gcs(self, file_path, retries=3):
        try:
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            blob_path = os.path.join(self.gcs_path, os.path.basename(file_path))
            blob = bucket.blob(blob_path)

            for attempt in range(1, retries + 1):
                try:
                    blob.upload_from_filename(file_path)
                    self._print_colored(f"☁️ Uploaded {file_path} → gs://{self.gcs_bucket}/{blob_path}", "info")
                    return
                except Exception as e:
                    self._print_colored(f"⚠️ Upload attempt {attempt} failed: {e}", "warn")
                    time.sleep(5)
            self._print_colored(f"❌ Failed to upload {file_path} after {retries} retries.", "error")

        except Exception as e:
            self._print_colored(f"❌ GCS upload setup failed: {e}", "error")


# Optional global instance
log = LogManager()

# Backward compatibility
def log_message(message, stage="general"):
    log.log_message(message, stage=stage)
