import argparse
import csv
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

from evaluate_od import run_od_evaluation


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_base = project_root / "configs" / "evaluate" / "evaluate_od.yaml"
    default_candidate = project_root / "configs" / "evaluate" / "evaluate_od_tensorrt.yaml"

    parser = argparse.ArgumentParser(description="Run OD evaluation twice and compare detector backends.")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=default_base,
        help=f"Baseline config, e.g. PyTorch CUDA (default: {default_base})",
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=default_candidate,
        help=f"Candidate config, e.g. TensorRT engine (default: {default_candidate})",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=project_root / "runs" / "od_compare",
        help="Directory where the comparison summary is saved.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="pytorch_cuda",
        help="Short label for the baseline run.",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="tensorrt",
        help="Short label for the candidate run.",
    )
    return parser.parse_args()


def _pct_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return float((baseline - candidate) / baseline * 100.0)


def _speedup_factor(baseline: float, candidate: float) -> float:
    if candidate <= 0:
        return 0.0
    return float(baseline / candidate)


def main():
    args = parse_args()
    baseline_cfg = OmegaConf.load(args.baseline_config.resolve())
    candidate_cfg = OmegaConf.load(args.candidate_config.resolve())

    baseline_metrics = run_od_evaluation(baseline_cfg, args.baseline_config.resolve())
    candidate_metrics = run_od_evaluation(candidate_cfg, args.candidate_config.resolve())

    comparison = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_run_dir": baseline_metrics.get("run_dir"),
        "candidate_run_dir": candidate_metrics.get("run_dir"),
        "baseline_detector_backend": baseline_metrics.get("detector_backend_actual", baseline_metrics.get("detector_backend_declared")),
        "candidate_detector_backend": candidate_metrics.get("detector_backend_actual", candidate_metrics.get("detector_backend_declared")),
        "avg_detect_ms_baseline": baseline_metrics.get("avg_detect_ms"),
        "avg_detect_ms_candidate": candidate_metrics.get("avg_detect_ms"),
        "detect_latency_improvement_percent": _pct_improvement(
            float(baseline_metrics.get("avg_detect_ms", 0.0)),
            float(candidate_metrics.get("avg_detect_ms", 0.0)),
        ),
        "detect_speedup_factor": _speedup_factor(
            float(baseline_metrics.get("avg_detect_ms", 0.0)),
            float(candidate_metrics.get("avg_detect_ms", 0.0)),
        ),
        "avg_pipeline_ms_baseline": baseline_metrics.get("avg_pipeline_ms"),
        "avg_pipeline_ms_candidate": candidate_metrics.get("avg_pipeline_ms"),
        "pipeline_improvement_percent": _pct_improvement(
            float(baseline_metrics.get("avg_pipeline_ms", 0.0)),
            float(candidate_metrics.get("avg_pipeline_ms", 0.0)),
        ),
        "pipeline_speedup_factor": _speedup_factor(
            float(baseline_metrics.get("avg_pipeline_ms", 0.0)),
            float(candidate_metrics.get("avg_pipeline_ms", 0.0)),
        ),
        "map50_delta": float(candidate_metrics.get("map50", 0.0)) - float(baseline_metrics.get("map50", 0.0)),
        "map50_95_delta": float(candidate_metrics.get("map50_95", 0.0)) - float(baseline_metrics.get("map50_95", 0.0)),
        "precision_delta": float(candidate_metrics.get("precision", 0.0)) - float(baseline_metrics.get("precision", 0.0)),
        "recall_delta": float(candidate_metrics.get("recall", 0.0)) - float(baseline_metrics.get("recall", 0.0)),
    }

    save_root = args.save_path.resolve()
    run_dir = save_root / time.strftime("run_%Y_%m_%d_%H_%M_%S")
    os.makedirs(run_dir, exist_ok=True)

    with open(run_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(run_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison.keys()))
        writer.writeheader()
        writer.writerow(comparison)

    print("\n===== OD Comparison Finished =====")
    print(f"Baseline ({args.baseline_label}) avg_detect_ms: {baseline_metrics.get('avg_detect_ms', 0.0):.3f}")
    print(f"Candidate ({args.candidate_label}) avg_detect_ms: {candidate_metrics.get('avg_detect_ms', 0.0):.3f}")
    print(f"Detect speedup factor: {comparison['detect_speedup_factor']:.3f}x")
    print(f"Detect latency improvement: {comparison['detect_latency_improvement_percent']:.2f}%")
    print(f"Saved comparison to: {run_dir}")


if __name__ == "__main__":
    main()
