# ==========================================
# A_0_2_log_algorithm_monitor.py
# LogManager v2 - Commit 2 (auto monitoring)
# ==========================================

import os
import json
import psutil
import shutil
import datetime
import time
import subprocess
from threading import Lock, Thread

class LogManager:
    """
    Centralized logging system with three modes:
      - operational: concise logs with periodic RAM/Disk monitoring
      - debug: detailed, frequent logs (blocks, resources)
      - silent: no console output, only JSON
    """

    def __init__(self, mode="operational", log_dir="./logs", log_interval_min=5):
        self.mode = mode
        self.log_dir = log_dir
        self.log_interval_min = log_interval_min
        self.log_file = os.path.join(log_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.lock = Lock()
        self._stop_flag = False
        os.makedirs(self.log_dir, exist_ok=True)

        self._print_colored(f"🟢 LogManager initialized in '{mode}' mode. Logs -> {self.log_file}", "info")

        # Start background monitor for resources
        if self.mode != "silent":
            self.monitor_thread = Thread(target=self._auto_monitor, daemon=True)
            self.monitor_thread.start()

    # ----------------------------------------
    # Public logging method
    # ----------------------------------------
    def log_message(self, message, stage="general", level="info"):
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "timestamp": timestamp,
            "stage": stage,
            "level": level,
            "message": message,
        }
        self._save_json(data)

        if self.mode != "silent":
            prefix = "🔹" if level == "info" else "⚠️" if level == "warn" else "❌"
            self._print_colored(f"[{timestamp}] {prefix} {message}", level)

    # ----------------------------------------
    # Resource snapshot (RAM, Disk, CPU, GPU)
    # ----------------------------------------
    def resources(self, auto=False):
        """Log system resource usage (RAM, Disk, CPU, GPU if available)."""
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
            # Log silently in background (avoid flooding console)
            data = {
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": "resources",
                "level": "info",
                "ram_used_gb": ram_used,
                "ram_total_gb": ram_total,
                "disk_used_gb": disk_used,
                "disk_total_gb": disk_total,
                "cpu_percent": cpu_percent,
                "gpu_info": gpu_info,
            }
            self._save_json(data)

    # ----------------------------------------
    # Background monitor thread
    # ----------------------------------------
    def _auto_monitor(self):
        """Runs periodic resource logging every N minutes."""
        while not self._stop_flag:
            time.sleep(self.log_interval_min * 60)
            if not self._stop_flag and self.mode in ["operational", "debug"]:
                self.resources(auto=True)

    # ----------------------------------------
    # GPU info (if nvidia-smi is available)
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
    # ANSI color printing
    # ----------------------------------------
    def _print_colored(self, msg, level="info"):
        colors = {
            "info": "\033[94m",     # Blue
            "warn": "\033[93m",     # Yellow
            "error": "\033[91m",    # Red
            "resources": "\033[92m" # Green
        }
        endc = "\033[0m"
        print(f"{colors.get(level, '')}{msg}{endc}")

    # ----------------------------------------
    # Save log line to JSONL
    # ----------------------------------------
    def _save_json(self, data):
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")

    # ----------------------------------------
    # Stop background monitor
    # ----------------------------------------
    def stop(self):
        self._stop_flag = True
        self._print_colored("🛑 LogManager background monitor stopped.", "warn")

    # ----------------------------------------
    # Change mode dynamically
    # ----------------------------------------
    def set_mode(self, mode):
        valid_modes = ["operational", "debug", "silent"]
        if mode not in valid_modes:
            self.log_message(f"Invalid mode '{mode}', keeping '{self.mode}'", level="warn")
            return
        self.mode = mode
        self.log_message(f"Switched logging mode to '{mode}'", level="info")

    # ----------------------------------------
    # Write summary (for upload later)
    # ----------------------------------------
    def summary(self, status="completed"):
        end_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        data = {"timestamp": end_time, "status": status}
        self._save_json(data)
        self._print_colored(f"🧾 Log summary recorded — status: {status}", "info")
        self.stop()


# Optional global instance
log = LogManager()

# Backward compatibility
def log_message(message, stage="general"):
    log.log_message(message, stage=stage)
