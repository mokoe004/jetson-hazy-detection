import subprocess
import threading
import re
import numpy as np


# --------------------------------------
# Tegrastats Monitor (Jetson)
# --------------------------------------
class TegrastatsMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.process = None

        self.gpu_usage = []
        self.power_usage = []
        self.ram_usage = []
        self.total_ram_mb = None

    def _monitor(self):
        self.process = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        while self.running:
            line = self.process.stdout.readline()
            if not line:
                continue

            # GPU Utilization
            gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_match:
                self.gpu_usage.append(int(gpu_match.group(1)))

            # GPU Power
            power_match = re.search(r'POM_5V_GPU (\d+)mW', line)
            if power_match:
                self.power_usage.append(int(power_match.group(1)))

            # RAM Usage
            ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
            if ram_match:
                used = int(ram_match.group(1))
                total = int(ram_match.group(2))

                self.ram_usage.append(used)

                if self.total_ram_mb is None:
                    self.total_ram_mb = total

        if self.process:
            self.process.terminate()
            self.process.wait()

    # --------------------------
    # Public API
    # --------------------------
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    # --------------------------
    # Metrics Helper
    # --------------------------
    def get_metrics(self):
        metrics = {}

        if self.gpu_usage:
            metrics["avg_gpu_utilization_percent"] = float(np.mean(self.gpu_usage))
            metrics["max_gpu_utilization_percent"] = float(np.max(self.gpu_usage))

        if self.power_usage:
            metrics["avg_gpu_power_watt"] = float(np.mean(self.power_usage) / 1000.0)
            metrics["max_gpu_power_watt"] = float(np.max(self.power_usage) / 1000.0)

        if self.ram_usage:
            avg_ram_mb = float(np.mean(self.ram_usage))
            metrics["avg_ram_usage_mb"] = avg_ram_mb
            metrics["max_ram_usage_mb"] = float(np.max(self.ram_usage))

            if self.total_ram_mb:
                metrics["avg_ram_usage_percent"] = float(
                    (avg_ram_mb / self.total_ram_mb) * 100.0
                )
                metrics["max_ram_usage_percent"] = float(
                    (np.max(self.ram_usage) / self.total_ram_mb) * 100.0
                )

        return metrics
