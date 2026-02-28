import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from tqdm import tqdm

from LCANet.lcanet import LCAnet
from models.aod_net import AODnet
from dataloaders import PairedDataset
from ssim_psnr_eval import ssim, psnr
from omegaconf import OmegaConf

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
        model = LCAnet().to(device)
    else:
        print("Model not known. Fallback to AODNet")
        model = AODnet().to(device)

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

    # --------------------------------------------------
    # 4) Dataset + Dataloaders
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),
    ])

    full_dataset = PairedDataset(cfg, transforms=transform)

    # Train val split 0.8 - 0.2
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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

        # -------- VALIDATION --------
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for i, (hazy, clear) in enumerate(val_loader):
                hazy, clear = hazy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
                prediction = model(hazy)

                total_psnr += psnr(prediction, clear)
                total_ssim += ssim(prediction, clear).item()

                # Beispielbild speichern (erste Val-Batch)
                if i == 0:
                    comparison = torch.cat([hazy[:1], prediction[:1], clear[:1]], dim=3)
                    utils.save_image(comparison, os.path.join(out_dir, f"epoch_{epoch+1:04d}.png"))

        avg_psnr = total_psnr / max(1, len(val_loader))
        avg_ssim = total_ssim / max(1, len(val_loader))

        # -------- Save LAST every epoch --------
        torch.save(model.state_dict(), last_path)

        # -------- Save BEST --------
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), best_path)

        # -------- CSV write --------
        lr = optimizer.param_groups[0]["lr"]
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
