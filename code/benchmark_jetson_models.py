import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.jetson_benchmark import TegrastatsMonitor
from models import AODNet, FFANet, LDNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FFANet, AODNet and LDNet on a Jetson device."
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=4,
        default=[1, 3, 256, 256],
        metavar=("N", "C", "H", "W"),
        help="Dummy input tensor shape.",
    )
    parser.add_argument("--runs", type=int, default=200, help="Measured inference runs per model.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per model.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use autocast fp16 on supported devices.",
    )
    parser.add_argument(
        "--enable-tegrastats",
        action="store_true",
        help="Collect Jetson tegrastats metrics during each benchmark.",
    )
    parser.add_argument(
        "--save-path",
        default="runs/benchmark_jetson_models",
        help="Directory where benchmark outputs are written.",
    )
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "FFANet":
        return FFANet()
    if model_name == "AODNet":
        return AODNet()
    if model_name == "LDNet":
        return LDNet()
    raise ValueError(f"Unknown model: {model_name}")


def load_model(model_name: str, device: torch.device):
    model = build_model(model_name)
    model = model.to(device)
    model.eval()
    return model


def benchmark_model(model_name, model, device, input_size, runs, warmup, use_fp16, enable_tegrastats):
    dummy = torch.randn(tuple(input_size), device=device)
    timings = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        with torch.no_grad():
            if use_fp16 and device.type != "cpu":
                with torch.autocast(device_type=device.type):
                    model(dummy)
            else:
                model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    monitor = TegrastatsMonitor() if enable_tegrastats else None
    if monitor:
        monitor.start()

    try:
        for _ in range(runs):
            start = time.perf_counter()
            with torch.no_grad():
                if use_fp16 and device.type != "cpu":
                    with torch.autocast(device_type=device.type):
                        model(dummy)
                else:
                    model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)
    finally:
        if monitor:
            monitor.stop()

    timings_np = np.array(timings, dtype=np.float64)
    mean_latency = float(timings_np.mean())
    metrics = {
        "model_name": model_name,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.median(timings_np)),
        "min_latency_ms": float(timings_np.min()),
        "max_latency_ms": float(timings_np.max()),
        "p95_latency_ms": float(np.percentile(timings_np, 95)),
        "p99_latency_ms": float(np.percentile(timings_np, 99)),
        "std_latency_ms": float(timings_np.std()),
        "fps": float(1000.0 / mean_latency),
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "parameters_millions": float(sum(p.numel() for p in model.parameters()) / 1e6),
    }

    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
        metrics["peak_gpu_memory_mb"] = float(peak_mem_bytes / (1024 ** 2))
        metrics["peak_gpu_memory_percent"] = float((peak_mem_bytes / total_mem_bytes) * 100.0)

    if monitor:
        jetson_metrics = monitor.get_metrics()
        metrics.update(jetson_metrics)
        if "avg_gpu_power_watt" in jetson_metrics:
            avg_latency_s = mean_latency / 1000.0
            metrics["energy_per_inference_joule"] = float(jetson_metrics["avg_gpu_power_watt"] * avg_latency_s)
            if jetson_metrics["avg_gpu_power_watt"] > 0:
                metrics["fps_per_watt"] = float(metrics["fps"] / jetson_metrics["avg_gpu_power_watt"])

    return metrics


def save_results(run_dir: Path, config, results):
    run_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=config, f=str(run_dir / "config.yaml"))

    with open(run_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    csv_columns = []
    for result in results:
        for key, value in result.items():
            if not isinstance(value, dict) and key not in csv_columns:
                csv_columns.append(key)

    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            flat_result = {key: value for key, value in result.items() if not isinstance(value, dict)}
            writer.writerow(flat_result)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    requested_device = torch.device(args.device)
    device = requested_device
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        device = torch.device("cpu")

    config = OmegaConf.create(
        {
            "benchmark": {
                "device": str(device),
                "requested_device": args.device,
                "input_size": list(args.input_size),
                "runs": args.runs,
                "warmup": args.warmup,
                "use_fp16": bool(args.fp16),
                "enable_tegrastats": bool(args.enable_tegrastats),
                "save_path": args.save_path,
            },
            "models": {
                "names": ["FFANet", "AODNet", "LDNet"],
            },
        }
    )

    timestamp = datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
    run_dir = (project_root / args.save_path / timestamp).resolve()

    results = []
    for model_name in ("FFANet", "AODNet", "LDNet"):
        print(f"\nBenchmarking {model_name} on {device} ...")
        model = load_model(model_name, device)
        metrics = benchmark_model(
            model_name=model_name,
            model=model,
            device=device,
            input_size=args.input_size,
            runs=args.runs,
            warmup=args.warmup,
            use_fp16=args.fp16,
            enable_tegrastats=args.enable_tegrastats,
        )
        results.append(metrics)
        print(
            f"{model_name}: "
            f"{metrics['mean_latency_ms']:.3f} ms | "
            f"{metrics['fps']:.2f} FPS | "
            f"{metrics['parameters_millions']:.3f} M params"
        )

    save_results(run_dir, config, results)

    print(f"\nSaved benchmark results to: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise
