import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from datasets import RTTSDataset, rtts_collate_fn
from detectors import YOLOv8Adapter
from evaluation.od_metrics import EvalSample, evaluate_detection
from utils import load_pretrained_dehazer


def _build_rtts_loader(cfg: DictConfig):
    dataset_cfg = cfg.dataset
    if str(dataset_cfg.name).lower() not in {"rtts"}:
        raise ValueError("OD evaluation currently supports dataset.name='RTTS'.")

    transform = transforms.Compose([transforms.ToTensor()])
    image_set = str(OmegaConf.select(cfg, "evaluation_od.image_set", default="test"))
    dataset = RTTSDataset(cfg, image_set=image_set, transforms=transform)

    subset = OmegaConf.select(cfg, "dataset.subset", default=False)
    if subset:
        dataset = Subset(dataset, range(int(subset)))

    batch_size = int(OmegaConf.select(cfg, "evaluation_od.batch_size", default=1))
    if batch_size != 1:
        print("Warning: batch_size>1 uses variable-size batches; processing remains per-image.")

    num_workers = int(OmegaConf.select(cfg, "evaluation_od.num_workers", default=0))
    pin_memory_cfg = bool(OmegaConf.select(cfg, "evaluation_od.pin_memory", default=True))
    pin_memory = pin_memory_cfg and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=rtts_collate_fn,
    )


def _dehaze_identity(image: torch.Tensor) -> torch.Tensor:
    return image


def _build_detector(cfg: DictConfig):
    detector_name = str(OmegaConf.select(cfg, "detector.name", default="yolov8")).lower()
    if detector_name != "yolov8":
        raise ValueError("Only detector.name='yolov8' is supported right now.")

    weights = OmegaConf.select(cfg, "detector.weights")
    if not weights:
        raise ValueError("Missing detector.weights in config.")

    detector_device = str(OmegaConf.select(cfg, "detector.device", default="cuda:0"))
    if detector_device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested for detector but not available. Falling back to CPU.")
        detector_device = "cpu"

    return YOLOv8Adapter(
        weights=str(weights),
        device=detector_device,
        conf=float(OmegaConf.select(cfg, "detector.conf", default=0.25)),
        iou=float(OmegaConf.select(cfg, "detector.iou", default=0.7)),
        imgsz=int(OmegaConf.select(cfg, "detector.imgsz", default=640)),
        max_det=int(OmegaConf.select(cfg, "detector.max_det", default=300)),
    )


def _to_eval_sample(pred, target, image_id: int) -> EvalSample:
    gt_boxes = target["boxes"].detach().cpu().to(torch.float32)
    gt_labels = target["labels"].detach().cpu().to(torch.int64)
    return EvalSample(
        pred_boxes=pred.boxes,
        pred_scores=pred.scores,
        pred_labels=pred.labels,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        image_id=image_id,
    )


def run_od_evaluation(cfg: DictConfig, config_path: Path) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    device_str = str(OmegaConf.select(cfg, "evaluation_od.device", default="cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    save_root = Path(str(OmegaConf.select(cfg, "evaluation_od.save_path", default="./runs/od_eval")))
    if not save_root.is_absolute():
        save_root = (project_root / save_root).resolve()
    run_dir = save_root / time.strftime("run_%Y_%m_%d_%H_%M_%S")
    os.makedirs(run_dir, exist_ok=True)

    cfg_to_save = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_to_save.evaluation_od = cfg_to_save.get("evaluation_od", {})
    cfg_to_save.evaluation_od["source_config_path"] = str(config_path.resolve())
    OmegaConf.save(cfg_to_save, run_dir / "run_config.yaml")

    use_dehazer = bool(OmegaConf.select(cfg, "evaluation_od.use_dehazer", default=True))
    dehazer = None
    dehazer_ckpt = "none"
    if use_dehazer:
        ckpt_raw = OmegaConf.select(cfg, "evaluation_od.dehazer_checkpoint_path")
        if not ckpt_raw:
            raise ValueError("Missing evaluation_od.dehazer_checkpoint_path in config.")
        dehazer, dehazer_ckpt = load_pretrained_dehazer(
            cfg=cfg,
            device=device,
            checkpoint_path=str(ckpt_raw),
            project_root=project_root,
            strict=True,
            print_info=True,
        )
    detector = _build_detector(cfg)
    loader = _build_rtts_loader(cfg)

    samples: List[EvalSample] = []
    dehaze_times = []
    detect_times = []
    dehazer_size = OmegaConf.select(cfg, "evaluation_od.dehazer_input_size", default=False)

    with torch.no_grad():
        image_counter = 0
        for images, targets in tqdm(loader, desc="OD Eval"):
            for image, target in zip(images, targets):
                image = image.to(device, non_blocking=True).unsqueeze(0)
                original_h, original_w = image.shape[-2:]

                t0 = time.perf_counter()
                dehaze_input = image
                if use_dehazer and dehazer_size:
                    size = int(dehazer_size)
                    dehaze_input = F.interpolate(
                        image, size=(size, size), mode="bilinear", align_corners=False
                    )

                if use_dehazer:
                    dehazed = dehazer(dehaze_input)
                    if dehazer_size:
                        dehazed = F.interpolate(
                            dehazed, size=(original_h, original_w), mode="bilinear", align_corners=False
                        )
                else:
                    dehazed = _dehaze_identity(dehaze_input)

                dehazed = dehazed.clamp(0.0, 1.0).squeeze(0).cpu()
                dehaze_times.append((time.perf_counter() - t0) * 1000.0)

                t1 = time.perf_counter()
                pred = detector.predict([dehazed])[0]
                detect_times.append((time.perf_counter() - t1) * 1000.0)

                samples.append(_to_eval_sample(pred, target, image_counter))
                image_counter += 1

    metrics = evaluate_detection(samples)
    metrics.update(
        {
            "dehazer_checkpoint": str(dehazer_ckpt),
            "dehazer_model": str(cfg.model.name) if use_dehazer else "none",
            "use_dehazer": use_dehazer,
            "detector_name": str(OmegaConf.select(cfg, "detector.name", default="yolov8")),
            "detector_weights": str(OmegaConf.select(cfg, "detector.weights", default="")),
            "device": str(device),
            "avg_dehaze_ms": float(sum(dehaze_times) / max(len(dehaze_times), 1)),
            "avg_detect_ms": float(sum(detect_times) / max(len(detect_times), 1)),
        }
    )

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("\n===== OD Evaluation Finished =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"mAP@0.5: {metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    print(f"Run saved to: {run_dir}")
    return metrics


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_config = project_root / "configs" / "evaluate_od.yaml"

    parser = argparse.ArgumentParser(description="Evaluate dehaze + object detection pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to OD evaluation config YAML (default: {default_config})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    run_od_evaluation(cfg, config_path)
