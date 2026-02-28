import subprocess
import threading
import time
import re
# --------------------------------------
# Tegrastats Monitor
# --------------------------------------
class TegrastatsMonitor:
    def __init__(self):
        self.running = False
        self.gpu_usage = []
        self.power_usage = []
        self.ram_usage = []

    def _monitor(self):
        process = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        while self.running:
            line = process.stdout.readline()

            gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_match:
                self.gpu_usage.append(int(gpu_match.group(1)))

            power_match = re.search(r'POM_5V_GPU (\d+)mW', line)
            if power_match:
                self.power_usage.append(int(power_match.group(1)))

            ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
            if ram_match:
                self.ram_usage.append(int(ram_match.group(1)))

        process.terminate()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()