import os
import sys
import time
from datetime import datetime
import json, csv
import platform

import torch
from omegaconf import OmegaConf
from torchvision import utils
import numpy as np
from tqdm import tqdm

from evaluation.ssim_psnr_eval import ssim, psnr
from evaluation.jetson_benchmark import TegrastatsMonitor

def calculate_psnr_ssim(
    model,
    dataloader,
    device,
    out_dir=None,
    save_example=True,
    filename_prefix="val"
):
    """
    Berechnet durchschnittlichen PSNR und SSIM über einen Dataloader.

    Args:
        model: PyTorch Modell
        dataloader: DataLoader
        device: torch.device
        out_dir: Optionaler Output-Ordner für Beispielbild
        save_example: Ob erstes Beispiel gespeichert werden soll
        filename_prefix: Prefix für gespeichertes Bild

    Returns:
        avg_psnr, avg_ssim
    """

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (hazy, clear) in enumerate(dataloader):
            hazy = hazy.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            prediction = model(hazy)

            total_psnr += psnr(prediction, clear)
            total_ssim += ssim(prediction, clear).item()
            num_batches += 1

            # Erstes Batch speichern
            if save_example and i == 0 and out_dir is not None:
                comparison = torch.cat(
                    [hazy[:1], prediction[:1], clear[:1]], dim=3
                )
                utils.save_image(
                    comparison,
                    os.path.join(out_dir, f"{filename_prefix}_example.png")
                )

    avg_psnr = total_psnr / max(1, num_batches)
    avg_ssim = total_ssim / max(1, num_batches)

    return avg_psnr, avg_ssim

def run_benchmark(cfg, model):

    device = torch.device(cfg.benchmark.device)
    model = model.to(device)
    model.eval()

    input_size = tuple(cfg.benchmark.input_size)
    dummy = torch.randn(input_size).to(device)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.benchmark.save_path, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    monitor = None
    if cfg.benchmark.jetson.enable_tegrastats:
        monitor = TegrastatsMonitor()

    # Warmup
    for _ in range(cfg.benchmark.warmup):
        with torch.no_grad():
            if cfg.benchmark.use_fp16:
                with torch.amp.autocast(cfg.benchmark.device):
                    model(dummy)
            else:
                model(dummy)

    torch.cuda.synchronize()

    timings = []

    if monitor:
        monitor.start()

    for _ in tqdm(range(cfg.benchmark.runs)):
        start = time.time()

        with torch.no_grad():
            if cfg.benchmark.use_fp16:
                with torch.cuda.amp.autocast():
                    model(dummy)
            else:
                model(dummy)

        torch.cuda.synchronize()
        end = time.time()

        timings.append((end - start) * 1000)

    if monitor:
        monitor.stop()

    timings = np.array(timings)

    metrics = {
        "mean_latency_ms": float(timings.mean()),
        "median_latency_ms": float(np.median(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "p99_latency_ms": float(np.percentile(timings, 99)),
        "std_latency_ms": float(timings.std()),
        "fps": float(1000.0 / timings.mean()),
    }

    if monitor:
        if monitor.gpu_usage:
            metrics["avg_gpu_utilization_percent"] = float(np.mean(monitor.gpu_usage))
        if monitor.power_usage:
            metrics["avg_gpu_power_watt"] = float(np.mean(monitor.power_usage) / 1000)
        if monitor.ram_usage:
            metrics["avg_ram_usage_mb"] = float(np.mean(monitor.ram_usage))

    system_info = {
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    metrics["system_info"] = system_info

    # Save JSON
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save CSV
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

    env = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    with open(os.path.join(run_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=4)

    print("\n===== Benchmark Finished =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    print(f"\nRun saved to: {run_dir}")

    return metrics