import argparse
import csv
import os
from pathlib import Path
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm

from datasets import ResideOTS
from evaluate_od import run_od_evaluation
from evaluation.evaluation import calculate_psnr_ssim
from omegaconf import OmegaConf

from utils import print_model_info, cfg_select_model


def configure_realtime_logging():
    # Ensure logs are flushed line-by-line in notebook subprocesses.
    for stream in (sys.stdout, sys.stderr):
        try:
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def _has_detector_cfg(cfg) -> bool:
    return OmegaConf.select(cfg, "detector.name", default=None) is not None


def _build_od_eval_cfg(cfg, checkpoint_path: str):
    """Build OD evaluation config with RTTS defaults for model selection during training."""
    od_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    od_cfg.dataset = od_cfg.get("dataset", {})
    od_cfg.dataset["name"] = "RTTS"
    od_cfg.dataset["root"] = "./datasets/RTTS/RTTS"
    od_cfg.dataset["return_bboxes"] = True
    default_subset = 300 if torch.cuda.is_available() else 100
    od_cfg.dataset["subset"] = int(OmegaConf.select(cfg, "training.od_eval_subset", default=default_subset))

    od_cfg.evaluation_od = od_cfg.get("evaluation_od", {})
    od_cfg.evaluation_od["use_dehazer"] = True
    od_cfg.evaluation_od["dehazer_checkpoint_path"] = str(checkpoint_path)
    od_cfg.evaluation_od["save_path"] = str(
        OmegaConf.select(cfg, "evaluation_od.save_path", default=os.path.join(cfg.model.save_path, "od_eval"))
    )
    od_cfg.evaluation_od["image_set"] = str(OmegaConf.select(cfg, "evaluation_od.image_set", default="test"))
    od_cfg.evaluation_od["batch_size"] = int(OmegaConf.select(cfg, "evaluation_od.batch_size", default=1))
    od_cfg.evaluation_od["num_workers"] = int(OmegaConf.select(cfg, "evaluation_od.num_workers", default=0))
    od_cfg.evaluation_od["pin_memory"] = bool(OmegaConf.select(cfg, "evaluation_od.pin_memory", default=True))
    od_cfg.evaluation_od["dehazer_input_size"] = OmegaConf.select(
        cfg, "evaluation_od.dehazer_input_size", default=False
    )

    return od_cfg


def _measure_dehaze_perf_ms(model, dataloader, device, max_batches: int = 10):
    model.eval()
    timings_ms = []

    with torch.no_grad():
        for i, (hazy, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            hazy = hazy.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(hazy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    if not timings_ms:
        return None, None

    avg_ms = float(sum(timings_ms) / len(timings_ms))
    fps = float(1000.0 / avg_ms) if avg_ms > 0 else None
    return avg_ms, fps


def train(cfg, config_path: Optional[Path] = None):
    # --------------------------------------------------
    # 1) Device + Run Directories
    # --------------------------------------------------
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.model.save_path, exist_ok=True)

    run_name = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(cfg.model.save_path, run_name)
    checkpoint_root = str(OmegaConf.select(cfg, "training.checkpoint_root", default="./checkpoints"))
    ckpt_dir = os.path.join(checkpoint_root, run_name)
    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nTraining on {device}")
    print(f"Run directory: {run_dir}\n")
    print(f"Checkpoint directory: {ckpt_dir}\n")

    # --------------------------------------------------
    # 2) Save Config
    # --------------------------------------------------
    cfg_path = os.path.join(run_dir, "run_config.yaml")
    OmegaConf.save(cfg, cfg_path)

    # --------------------------------------------------
    # 3) Model + Optimizer
    # --------------------------------------------------
    model = cfg_select_model(cfg, cfg.training.device if torch.cuda.is_available() else "cpu")
    print_model_info(model)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # PSNR should increase
        factor=0.5,
        patience=5,
        threshold=0.01,
        cooldown=0,
        min_lr=1e-6,
    )

    # --------------------------------------------------
    # 4) Dataset + Dataloaders
    # --------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = ResideOTS(cfg, transforms=transform)
    if cfg.dataset.subset:
        dataset = Subset(dataset, range(cfg.dataset.subset))

    # Train val split 0.8 - 0.2
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n", flush=True)

    # --------------------------------------------------
    # 5) CSV Logger init
    # --------------------------------------------------
    csv_path = os.path.join(run_dir, "training_log.csv")
    csv_header = [
        "epoch",
        "train_loss",
        "psnr",
        "ssim",
        "avg_dehaze_ms",
        "dehaze_fps",
        "od_map50",
        "od_map50_95",
        "od_avg_dehaze_ms",
        "selection_score",
        "lr",
        "epoch_time_sec",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # --------------------------------------------------
    # 6) Training Loop (best + last)
    # --------------------------------------------------
    selection_metric = str(OmegaConf.select(cfg, "training.selection_metric", default="psnr")).lower()
    od_eval_enabled = _has_detector_cfg(cfg)
    od_eval_every = int(OmegaConf.select(cfg, "training.od_eval_every", default=5))
    alpha_latency = float(OmegaConf.select(cfg, "training.alpha_latency", default=0.001))
    perf_batches = int(OmegaConf.select(cfg, "training.perf_batches", default=10))

    if selection_metric not in {"psnr", "map50", "hybrid"}:
        raise ValueError("training.selection_metric must be one of: psnr, map50, hybrid")

    if not od_eval_enabled and selection_metric in {"map50", "hybrid"}:
        raise ValueError("selection_metric uses OD metrics, but no detector config is provided.")

    best_psnr = float("-inf")
    best_map50 = float("-inf")
    best_score = float("-inf")
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    best_psnr_path = os.path.join(ckpt_dir, "best_psnr_model.pth")
    best_map50_path = os.path.join(ckpt_dir, "best_map50_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    epoch_bar = tqdm(
        range(cfg.training.epochs),
        desc="Training",
        unit="epoch",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        t0 = time.time()

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for hazy, clear in train_loader:
            hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)

            optimizer.zero_grad()
            prediction = model(hazy)
            loss = criterion(prediction, clear)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        avg_psnr, avg_ssim = calculate_psnr_ssim(
            model,
            val_loader,
            device=device,
            out_dir=out_dir,
            save_example=True,
            filename_prefix=f"train_epoch{epoch:03d}",
        )
        avg_dehaze_ms, dehaze_fps = _measure_dehaze_perf_ms(
            model=model,
            dataloader=val_loader,
            device=device,
            max_batches=perf_batches,
        )

        scheduler.step(avg_psnr)

        # -------- Save LAST + per-epoch checkpoint --------
        torch.save(model.state_dict(), last_path)
        epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)

        # -------- Optional OD eval (RTTS) --------
        od_map50 = None
        od_map50_95 = None
        od_avg_dehaze_ms = None
        if od_eval_enabled and ((epoch + 1) % od_eval_every == 0 or (epoch + 1) == cfg.training.epochs):
            od_eval_cfg = _build_od_eval_cfg(cfg, checkpoint_path=last_path)
            od_metrics = run_od_evaluation(od_eval_cfg, config_path or Path("train_runtime"))
            od_map50 = float(od_metrics["map50"])
            od_map50_95 = float(od_metrics["map50_95"])
            od_avg_dehaze_ms = float(od_metrics["avg_dehaze_ms"])

        # -------- Track BEST --------
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), best_psnr_path)

        if od_map50 is not None and od_map50 > best_map50:
            best_map50 = od_map50
            torch.save(model.state_dict(), best_map50_path)

        if selection_metric == "psnr":
            selection_score = float(avg_psnr)
        elif selection_metric == "map50":
            selection_score = float(od_map50) if od_map50 is not None else float("-inf")
        else:
            if od_map50 is None or od_avg_dehaze_ms is None:
                selection_score = float("-inf")
            else:
                selection_score = float(od_map50 - alpha_latency * od_avg_dehaze_ms)

        if selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), best_path)

        # -------- CSV write --------
        lr = scheduler.optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    avg_train_loss,
                    avg_psnr,
                    avg_ssim,
                    avg_dehaze_ms,
                    dehaze_fps,
                    od_map50,
                    od_map50_95,
                    od_avg_dehaze_ms,
                    selection_score,
                    lr,
                    epoch_time,
                ]
            )

        # -------- print each epoch line --------
        metric_msg = ""
        if od_map50 is not None:
            metric_msg = (
                f" | mAP50: {od_map50:.4f} | mAP50-95: {od_map50_95:.4f} | Dehaze: {od_avg_dehaze_ms:.1f}ms"
            )

        tqdm.write(
            f"Epoch [{epoch + 1}/{cfg.training.epochs}] | "
            f"Loss: {avg_train_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f} | "
            f"Perf: {avg_dehaze_ms:.2f}ms ({dehaze_fps:.1f} FPS) | "
            f"Score({selection_metric}): {selection_score:.4f}{metric_msg} | "
            f"LR: {lr:.2e} | {epoch_time:.1f}s"
        )

    print("\nTraining finished.")
    print(f"Best model: {best_path} (score={best_score:.4f}, metric={selection_metric})")
    print(f"Best PSNR model: {best_psnr_path} (PSNR={best_psnr:.2f})")
    if best_map50 > float("-inf"):
        print(f"Best mAP50 model: {best_map50_path} (mAP50={best_map50:.4f})")
    print(f"CSV log:   {csv_path}")
    print(f"Last:      {last_path}")
    print(f"Outputs:   {out_dir}")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_config = project_root / "configs" / "config.yaml"

    parser = argparse.ArgumentParser(description="Train dehazing model with YAML config.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to training config YAML (default: {default_config})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_realtime_logging()
    args = parse_args()
    config_path = args.config.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    train(cfg, config_path=config_path)
