import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_config = project_root / "configs" / "evaluate" / "evaluate_od.yaml"

    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO weights to TensorRT engine format.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to evaluation config YAML (default: {default_config})",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Override detector weights path. Must point to a PyTorch .pt model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Export image size. Defaults to detector.imgsz from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Export device passed to Ultralytics, e.g. '0' or 'dla:0'.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size baked into the engine export.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export TensorRT engine with FP16 tactics when supported.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes during export.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="TensorRT workspace size in GiB.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 export. Requires suitable calibration data.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML used by Ultralytics for INT8 calibration.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of calibration data to use for INT8 export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config.resolve())

    weights = args.weights or str(OmegaConf.select(cfg, "detector.weights"))
    if not weights:
        raise ValueError("No detector weights provided. Set detector.weights in the config or pass --weights.")
    if not str(weights).lower().endswith(".pt"):
        raise ValueError(f"TensorRT export expects a PyTorch .pt model, got: {weights}")

    imgsz = int(args.imgsz or OmegaConf.select(cfg, "detector.imgsz", default=640))

    from ultralytics import YOLO

    model = YOLO(weights)
    export_kwargs = {
        "format": "engine",
        "imgsz": imgsz,
        "device": args.device,
        "batch": int(args.batch),
        "half": bool(args.half),
        "dynamic": bool(args.dynamic),
    }
    if args.workspace is not None:
        export_kwargs["workspace"] = float(args.workspace)
    if args.int8:
        export_kwargs["int8"] = True
    if args.data:
        export_kwargs["data"] = args.data
    if args.fraction is not None:
        export_kwargs["fraction"] = float(args.fraction)

    engine_path = model.export(**export_kwargs)
    print("===== YOLO TensorRT Export Finished =====")
    print(f"Source weights: {weights}")
    print(f"Exported engine: {engine_path}")
    print("Update your evaluation config:")
    print(f"detector.weights: \"{engine_path}\"")


if __name__ == "__main__":
    main()
