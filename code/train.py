import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, utils
from tqdm import tqdm

from models.lca_net import LCANet
from models.aod_net import AODnet
from dataloaders import ResideOTS
from evaluation.evaluation import calculate_psnr_ssim
from omegaconf import OmegaConf

from utils import print_model_info


def train(cfg):
    # --------------------------------------------------
    # 1) Device + Run Directories
    # --------------------------------------------------
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.model.save_path, exist_ok=True)

    run_dir = os.path.join(cfg.model.save_path, time.strftime("run_%Y%m%d_%H%M%S"))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🚀 Training auf {device}")
    print(f"📁 Run directory: {run_dir}\n")

    # --------------------------------------------------
    # 2) Save Config
    # --------------------------------------------------
    cfg_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_path)

    # --------------------------------------------------
    # 3) Model + Optimizer
    # --------------------------------------------------
    if cfg.model.name == "AODnet":
        model = AODnet().to(device)
    elif cfg.model.name == "LCAnet":
        model = LCANet().to(device)
    else:
        print("Model not known. Fallback to AODNet")
        model = AODnet().to(device)
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
        weight_decay=cfg.training.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # PSNR soll steigen
        factor=0.5,  # LR halbieren
        patience=5,  # 5 Epochen ohne Verbesserung
        threshold=0.01,  # min. 0.01 dB Verbesserung
        cooldown=0,
        min_lr=1e-6,
    )

    # --------------------------------------------------
    # 4) Dataset + Dataloaders
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),
    ])

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
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    # --------------------------------------------------
    # 5) CSV Logger init
    # --------------------------------------------------
    csv_path = os.path.join(run_dir, "training_log.csv")
    csv_header = ["epoch", "train_loss", "psnr", "ssim", "lr", "epoch_time_sec"]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # --------------------------------------------------
    # 6) Training Loop (best + last)
    # --------------------------------------------------
    best_psnr = float("-inf")
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    epoch_bar = tqdm(range(cfg.training.epochs), desc="Training", unit="epoch")

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
            filename_prefix=f"train_epoch{epoch:03d}"
            )

        scheduler.step(avg_psnr)

        # -------- Save LAST every epoch --------
        torch.save(model.state_dict(), last_path)

        # -------- Save BEST --------
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), best_path)

        # -------- CSV write --------
        lr = scheduler.optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_psnr, avg_ssim, lr, epoch_time])

        # -------- print each epoch line --------
        tqdm.write(
            f"Epoch [{epoch+1}/{cfg.training.epochs}] | "
            f"Loss: {avg_train_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f} | "
            f"LR: {lr:.2e} | {epoch_time:.1f}s"
        )

    print("\n✅ Training abgeschlossen.")
    print(f"⭐ Best model: {best_path} (PSNR={best_psnr:.2f})")
    print(f"🧾 CSV log:   {csv_path}")
    print(f"🧠 Last:      {last_path}")
    print(f"🖼️ Outputs:   {out_dir}")
