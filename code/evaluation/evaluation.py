import os
import torch
from torchvision import utils

from evaluation.ssim_psnr_eval import ssim, psnr

def calculate_psnr_ssim(
    model,
    dataloader,
    device,
    out_dir=None,
    save_example=True,
    filename_prefix="val"
):
    """
    Berechnet durchschnittlichen PSNR und SSIM über einen Dataloader.

    Args:
        model: PyTorch Modell
        dataloader: DataLoader
        device: torch.device
        out_dir: Optionaler Output-Ordner für Beispielbild
        save_example: Ob erstes Beispiel gespeichert werden soll
        filename_prefix: Prefix für gespeichertes Bild

    Returns:
        avg_psnr, avg_ssim
    """

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (hazy, clear) in enumerate(dataloader):
            hazy = hazy.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            prediction = model(hazy)

            total_psnr += psnr(prediction, clear)
            total_ssim += ssim(prediction, clear).item()
            num_batches += 1

            # Erstes Batch speichern
            if save_example and i == 0 and out_dir is not None:
                comparison = torch.cat(
                    [hazy[:1], prediction[:1], clear[:1]], dim=3
                )
                utils.save_image(
                    comparison,
                    os.path.join(out_dir, f"{filename_prefix}_example.png")
                )

    avg_psnr = total_psnr / max(1, num_batches)
    avg_ssim = total_ssim / max(1, num_batches)

    return avg_psnr, avg_ssim


