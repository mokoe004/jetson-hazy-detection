import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from train import configure_realtime_logging, train
from utils import cfg_select_model


DEFAULT_VARIANTS: List[Dict] = [
    {
        "name": "legacy_gaussian_b3",
        "model": {
            "base_channels": 3,
            "attention_variant": "legacy_gaussian",
            "num_attention_peaks": 1,
            "use_input_edge": False,
            "use_channel_gate": False,
            "alpha_init": 0.5,
            "sigma_scale": 0.3,
        },
    },
    {
        "name": "od_guided_b3",
        "model": {
            "base_channels": 3,
            "attention_variant": "od_guided",
            "num_attention_peaks": 4,
            "use_input_edge": True,
            "use_channel_gate": False,
            "alpha_init": 0.35,
            "sigma_scale": 0.4,
        },
    },
    {
        "name": "od_guided_b6_gate",
        "model": {
            "base_channels": 6,
            "attention_variant": "od_guided",
            "num_attention_peaks": 4,
            "use_input_edge": True,
            "use_channel_gate": True,
            "channel_gate_reduction": 8,
            "alpha_init": 0.35,
            "sigma_scale": 0.4,
            "spatial_hidden_channels": 8,
        },
    },
]


def _count_model_params(cfg) -> Dict[str, int]:
    model = cfg_select_model(cfg, device="cpu")
    return {
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def _read_training_log(csv_path: Path) -> Dict[str, float]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    def _best_float(key: str):
        values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
        return max(values) if values else None

    last = rows[-1]
    return {
        "epochs_ran": int(last["epoch"]),
        "last_train_loss": float(last["train_loss"]),
        "last_psnr": float(last["psnr"]),
        "last_ssim": float(last["ssim"]),
        "last_avg_dehaze_ms": float(last["avg_dehaze_ms"]),
        "last_dehaze_fps": float(last["dehaze_fps"]),
        "last_od_map50": float(last["od_map50"]) if last["od_map50"] else None,
        "last_od_map50_95": float(last["od_map50_95"]) if last["od_map50_95"] else None,
        "best_psnr_logged": _best_float("psnr"),
        "best_od_map50_logged": _best_float("od_map50"),
    }


def _build_variant_cfg(base_cfg, args, variant: Dict, output_root: Path):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.training.device = "cpu"
    cfg.training.epochs = int(args.epochs)
    cfg.training.batch_size = int(args.batch_size)
    cfg.training.num_workers = 0
    cfg.training.od_eval_every = int(args.epochs)
    cfg.training.od_eval_subset = int(args.od_subset)
    cfg.training.perf_batches = min(4, int(args.od_subset))
    cfg.training.checkpoint_root = str((output_root / "checkpoints" / variant["name"]).as_posix())
    cfg.training.selection_metric = "hybrid"
    cfg.training.early_stopping.enabled = False

    cfg.dataset.subset = int(args.train_subset)
    cfg.dataset.return_bboxes = False

    cfg.model.name = "AODNetDepthwiseSpatial"
    cfg.model.save_path = str((output_root / "train_runs" / variant["name"]).as_posix())
    for key, value in variant["model"].items():
        cfg.model[key] = value

    cfg.detector.device = "cpu"
    cfg.detector.imgsz = int(args.detector_imgsz)

    cfg.evaluation_od.device = "cpu"
    cfg.evaluation_od.save_path = str((output_root / "od_eval_runs" / variant["name"]).as_posix())
    cfg.evaluation_od.batch_size = 1
    cfg.evaluation_od.num_workers = 0
    cfg.evaluation_od.pin_memory = False
    cfg.evaluation_od.dehazer_input_size = int(args.dehazer_input_size)
    cfg.evaluation_od.visualization = {"enabled": False}
    return cfg


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a small AODNetDepthwiseSpatial ablation on CPU.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "train" / "train_aod_spatial.yaml",
        help="Base training config used as the starting point for all ablations.",
    )
    parser.add_argument("--train-subset", type=int, default=64, help="Number of RESIDE-OTS samples used for training.")
    parser.add_argument("--od-subset", type=int, default=30, help="Number of RTTS images used for OD evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per variant.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--detector-imgsz", type=int, default=512, help="YOLO inference size.")
    parser.add_argument("--dehazer-input-size", type=int, default=256, help="Fixed dehazer input size for OD eval.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "runs" / "ablation" / "aod_spatial",
        help="Directory where ablation artifacts are stored.",
    )
    return parser.parse_args()


def main():
    configure_realtime_logging()
    args = parse_args()
    base_cfg_path = args.config.resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    base_cfg = OmegaConf.load(base_cfg_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in DEFAULT_VARIANTS:
        print(f"\n=== Running variant: {variant['name']} ===")
        cfg = _build_variant_cfg(base_cfg, args, variant, output_root)
        params = _count_model_params(cfg)
        train_result = train(cfg, config_path=base_cfg_path)
        log_summary = _read_training_log(Path(train_result["csv_path"]))

        result = {
            "variant": variant["name"],
            "attention_variant": str(cfg.model.attention_variant),
            "base_channels": int(cfg.model.base_channels),
            **params,
            **log_summary,
            "run_dir": str(train_result["run_dir"]),
            "best_model_path": str(train_result["best_model_path"]),
            "best_map50_model_path": str(train_result["best_map50_model_path"])
            if train_result["best_map50_model_path"]
            else None,
        }
        results.append(result)

    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved ablation summary to: {json_path}")
    print(f"Saved ablation CSV to: {csv_path}")


if __name__ == "__main__":
    main()
