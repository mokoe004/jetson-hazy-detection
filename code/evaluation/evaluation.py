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

from utils import cfg_select_model

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
            if save_example and i == 5 and out_dir is not None:
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

def run_benchmark(cfg):

    device = torch.device(cfg.benchmark.device)
    model = cfg_select_model(cfg, device.type)
    model.eval()

    input_size = tuple(cfg.benchmark.input_size)
    dummy = torch.randn(input_size).to(device)

    # Reset GPU memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ----------------------------------
    # Create run directory
    # ----------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.benchmark.save_path, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    monitor = None
    if cfg.benchmark.jetson.enable_tegrastats:
        monitor = TegrastatsMonitor()

    # ----------------------------------
    # Warmup
    # ----------------------------------
    for _ in range(cfg.benchmark.warmup):
        with torch.no_grad():
            if cfg.benchmark.use_fp16:
                with torch.autocast(device_type=device.type):
                    model(dummy)
            else:
                model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []

    # ----------------------------------
    # Start Monitoring
    # ----------------------------------
    if monitor:
        monitor.start()

    # ----------------------------------
    # Benchmark Loop
    # ----------------------------------
    for _ in tqdm(range(cfg.benchmark.runs)):
        start = time.time()

        with torch.no_grad():
            if cfg.benchmark.use_fp16:
                with torch.autocast(device_type=device.type):
                    model(dummy)
            else:
                model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()
        timings.append((end - start) * 1000)

    if monitor:
        monitor.stop()

    timings = np.array(timings)

    # ----------------------------------
    # Core Latency Metrics
    # ----------------------------------
    mean_latency = float(timings.mean())

    metrics = {
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.median(timings)),
        "min_latency_ms": float(timings.min()),
        "max_latency_ms": float(timings.max()),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "p99_latency_ms": float(np.percentile(timings, 99)),
        "std_latency_ms": float(timings.std()),
        "fps": float(1000.0 / mean_latency),
    }

    # ----------------------------------
    # GPU Memory Metrics
    # ----------------------------------
    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)

        total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
        total_mem_mb = total_mem_bytes / (1024 ** 2)

        metrics["peak_gpu_memory_mb"] = float(peak_mem_mb)
        metrics["peak_gpu_memory_percent"] = float(
            (peak_mem_mb / total_mem_mb) * 100.0
        )

    # ----------------------------------
    # Parameter Metrics
    # ----------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    metrics["parameters_total"] = int(total_params)
    metrics["parameters_trainable"] = int(trainable_params)
    metrics["parameters_millions"] = float(total_params / 1e6)

    # ----------------------------------
    # Jetson Power / RAM Metrics
    # ----------------------------------
    if monitor:
        jetson_metrics = monitor.get_metrics()
        metrics.update(jetson_metrics)

        # Energy per inference
        if "avg_gpu_power_watt" in jetson_metrics:
            avg_power = jetson_metrics["avg_gpu_power_watt"]
            avg_latency_s = mean_latency / 1000.0
            energy = avg_power * avg_latency_s

            metrics["energy_per_inference_joule"] = float(energy)

            if avg_power > 0:
                metrics["fps_per_watt"] = float(metrics["fps"] / avg_power)

    # ----------------------------------
    # System Info
    # ----------------------------------
    metrics["system_info"] = {
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
    }

    # ----------------------------------
    # Save JSON
    # ----------------------------------
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ----------------------------------
    # Save CSV (nur flache Werte)
    # ----------------------------------
    flat_metrics = {
        k: v for k, v in metrics.items() if not isinstance(v, dict)
    }

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(flat_metrics.keys())
        writer.writerow(flat_metrics.values())

    # ----------------------------------
    # Environment
    # ----------------------------------
    env = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    with open(os.path.join(run_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=4)

    # ----------------------------------
    # Console Output (format safe)
    # ----------------------------------
    print("\n===== Benchmark Finished =====")

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    print(f"\nRun saved to: {run_dir}")

    return metrics
