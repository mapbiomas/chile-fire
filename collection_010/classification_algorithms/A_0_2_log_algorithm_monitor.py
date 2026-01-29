# ==========================================
# A_0_2_log_algorithm_monitor.py
# New LogManager (commit 1)
# ==========================================

import os
import json
import psutil
import shutil
import datetime
import time
from threading import Lock

class LogManager:
    """
    Centralized logging system with three modes:
    - operational: concise, human-readable with resource summary
    - debug: detailed, more frequent logs
    - silent: no console output, only JSON file
    """

    def __init__(self, mode="operational", log_dir="./logs"):
        self.mode = mode
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.lock = Lock()
        os.makedirs(self.log_dir, exist_ok=True)
        self._print_colored(f"🟢 LogManager initialized in '{mode}' mode. Logs -> {self.log_file}", "info")

    # ----------------------------------------
    # Public logging method (backward-compatible)
    # ----------------------------------------
    def log_message(self, message, stage="general", level="info"):
        """Record an event (console + JSON)."""
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
    # Resource snapshot (RAM, Disk)
    # ----------------------------------------
    def resources(self):
        """Log system resource usage (RAM, disk)."""
        vm = psutil.virtual_memory()
        du = shutil.disk_usage('/')
        ram_used = round(vm.used / 1024**3, 2)
        ram_total = round(vm.total / 1024**3, 2)
        disk_used = round(du.used / 1024**3, 2)
        disk_total = round(du.total / 1024**3, 2)

        msg = f"💾 RAM: {ram_used} GB / {ram_total} GB | Disk: {disk_used} GB / {disk_total} GB"
        self.log_message(msg, stage="resources")

    # ----------------------------------------
    # Print with ANSI colors
    # ----------------------------------------
    def _print_colored(self, msg, level="info"):
        colors = {
            "info": "\033[94m",   # Blue
            "warn": "\033[93m",   # Yellow
            "error": "\033[91m",  # Red
            "resources": "\033[92m"  # Green
        }
        endc = "\033[0m"
        print(f"{colors.get(level, '')}{msg}{endc}")

    # ----------------------------------------
    # Save log as JSON line
    # ----------------------------------------
    def _save_json(self, data):
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")

    # ----------------------------------------
    # Change logging mode dynamically
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


# Optional global instance
log = LogManager()

# Backward compatibility
def log_message(message, stage="general"):
    log.log_message(message, stage=stage)
