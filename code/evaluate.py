import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from dataloaders import PairedDataset, ResideOTS
from evaluation.evaluation import calculate_psnr_ssim
from utils import load_pretrained_dehazer


def _select_dataset(cfg: DictConfig, transform):
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name in {"reside_ots", "resideots"}:
        return ResideOTS(cfg, transforms=transform)
    if dataset_name in {"paired", "paireddataset"}:
        return PairedDataset(cfg, transforms=transform)
    raise ValueError(f"Unsupported dataset.name '{cfg.dataset.name}' for evaluation.")


def _build_eval_loader(cfg: DictConfig):
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = _select_dataset(cfg, transform)
    if cfg.dataset.subset:
        dataset = Subset(dataset, range(cfg.dataset.subset))

    split_mode = str(OmegaConf.select(cfg, "evaluation.split", default="val")).lower()
    val_split = float(OmegaConf.select(cfg, "evaluation.val_split", default=0.2))
    seed = int(OmegaConf.select(cfg, "evaluation.seed", default=42))

    if split_mode == "full":
        eval_dataset = dataset
    elif split_mode == "train":
        val_size = max(1, int(val_split * len(dataset)))
        train_size = max(0, len(dataset) - val_size)
        if train_size == 0:
            raise ValueError("Train split is empty. Reduce 'evaluation.val_split' or use a larger dataset.")
        torch.manual_seed(seed)
        train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
        eval_dataset = train_dataset
    elif split_mode == "val":
        val_size = max(1, int(val_split * len(dataset)))
        train_size = max(0, len(dataset) - val_size)
        if val_size == 0:
            raise ValueError("Validation split is empty. Increase dataset size or val_split.")
        torch.manual_seed(seed)
        _, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        eval_dataset = val_dataset
    else:
        raise ValueError("evaluation.split must be one of: 'val', 'train', 'full'.")

    batch_size = int(
        OmegaConf.select(
            cfg,
            "evaluation.batch_size",
            default=OmegaConf.select(cfg, "training.batch_size", default=1),
        )
    )
    num_workers = int(
        OmegaConf.select(
            cfg,
            "evaluation.num_workers",
            default=OmegaConf.select(cfg, "training.num_workers", default=0),
        )
    )
    pin_memory = bool(OmegaConf.select(cfg, "evaluation.pin_memory", default=True))

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, len(eval_dataset), len(dataset)


def evaluate(cfg: DictConfig, config_path: Path) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    device_str = OmegaConf.select(cfg, "evaluation.device", default=cfg.training.device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    checkpoint_raw = OmegaConf.select(cfg, "evaluation.checkpoint_path")
    if not checkpoint_raw:
        raise ValueError("Missing 'evaluation.checkpoint_path' in config.")

    save_root = OmegaConf.select(cfg, "evaluation.save_path", default="./runs/eval")
    run_root = Path(save_root)
    if not run_root.is_absolute():
        run_root = (project_root / run_root).resolve()
    run_dir = run_root / time.strftime("run_%Y_%m_%d_%H_%M_%S")
    outputs_dir = run_dir / "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    model, checkpoint_path = load_pretrained_dehazer(
        cfg=cfg,
        device=device,
        checkpoint_path=str(checkpoint_raw),
        project_root=project_root,
        strict=True,
        print_info=True,
    )

    cfg_to_save = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_to_save.evaluation = cfg_to_save.get("evaluation", {})
    cfg_to_save.evaluation["resolved_checkpoint_path"] = str(checkpoint_path)
    cfg_to_save.evaluation["source_config_path"] = str(config_path.resolve())
    OmegaConf.save(cfg_to_save, run_dir / "run_config.yaml")

    loader, eval_count, total_count = _build_eval_loader(cfg)
    print(f"Evaluation on {eval_count}/{total_count} samples ({OmegaConf.select(cfg, 'evaluation.split', default='val')})")

    avg_psnr, avg_ssim = calculate_psnr_ssim(
        model=model,
        dataloader=loader,
        device=device,
        out_dir=str(outputs_dir),
        save_example=bool(OmegaConf.select(cfg, "evaluation.save_example", default=True)),
        filename_prefix=str(OmegaConf.select(cfg, "evaluation.filename_prefix", default="eval")),
    )

    metrics = {
        "psnr": float(avg_psnr),
        "ssim": float(avg_ssim),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "samples_evaluated": int(eval_count),
        "samples_total_after_subset": int(total_count),
        "split": str(OmegaConf.select(cfg, "evaluation.split", default="val")),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("\n===== Evaluation Finished =====")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"Run saved to: {run_dir}")

    return metrics


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_config = project_root / "configs" / "config.yaml"

    parser = argparse.ArgumentParser(description="Evaluate dehazing model with YAML config.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to evaluation config YAML (default: {default_config})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    evaluate(cfg, config_path)
