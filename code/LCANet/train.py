import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# =========================
# Dein Model importieren
# =========================
from lca_net import LCANet
from dataloader import DehazingDataset

# =========================
# PSNR Funktion
# =========================
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# =========================
# Training Funktion
# =========================
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------
    # Dataset & Loader
    # -------------------------
    train_dataset = DehazingDataset(
        root_dir="../../datasets/hazy-no-bb",
        subset="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # Model
    # -------------------------
    model = LCANet(out_act="sigmoid").to(device)

    # -------------------------
    # Loss & Optimizer
    # -------------------------
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------
    # Training Loop
    # -------------------------
    epochs = 50
    best_psnr = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        running_psnr = 0

        loop = tqdm(train_loader)

        for foggy, clear in loop:

            foggy = foggy.to(device)
            clear = clear.to(device)

            # Forward
            pred = model(foggy)

            loss = criterion(pred, clear)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            running_loss += loss.item()
            running_psnr += compute_psnr(pred.detach(), clear).item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        avg_psnr = running_psnr / len(train_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

        # -------------------------
        # Save Best Model
        # -------------------------
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), "best_model.pth")
            print("✔ Best model saved")

    print("Training Finished.")


if __name__ == "__main__":
    train()