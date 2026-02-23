import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from lca_net import LCANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Modell laden
# -------------------------------------------------
model = LCANet(out_act="sigmoid").to(device)
model.eval()

# -------------------------------------------------
# Dummy Input (0–1 Range!)
# -------------------------------------------------
x = torch.rand(1, 3, 512, 512, device=device)

# -------------------------------------------------
# 🔥 INFERENCE
# -------------------------------------------------
with torch.no_grad():
    y = model(x)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
mse_val = F.mse_loss(y, x)

psnr_val = 20 * torch.log10(torch.tensor(1.0, device=device)) \
           - 10 * torch.log10(mse_val)

ssim_val = ssim(y, x, data_range=1.0, size_average=True)

print("Output shape:", y.shape)
print("MSE  :", mse_val.item())
print("PSNR :", psnr_val.item())
print("SSIM :", ssim_val.item())