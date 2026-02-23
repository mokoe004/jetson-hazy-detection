import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from aodnet import aodnet
from PIL import Image
from utils import ssim
import numpy as np
from torchvision.utils import save_image

root_dir = 'path_to_root_directory'
save_dir = 'path_to_save_directory'

os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Dataset and DataLoader
test_dataset = DehazingDataset(root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Loading
try:
    model = torch.load(os.path.join(root_dir, 'Dehazed-Detection/AOD-Net/aodnet.pth')).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Loss Function
loss_fn = nn.MSELoss()
running_loss = 0.0
total_ssim = 0.0
num_samples = 0

with torch.no_grad():
    for data in test_loader:
        original_images = data['original_image'].to(device)
        foggy_images = data['foggy_image'].to(device)
        filenames = data['filename']

        outputs = model(foggy_images)
        loss = loss_fn(outputs, original_images)

        outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
        original_images_np = original_images.permute(0, 2, 3, 1).cpu().numpy()

        #calculating batch ssim score
        batch_ssim = np.mean([ssim(out, orig) for out, orig in zip(outputs_np, original_images_np)])

        running_loss += loss.item() * original_images.size(0)
        total_ssim += batch_ssim * original_images.size(0)
        num_samples += original_images.size(0)

        # Save dehazed images
        for output, filename in zip(outputs, filenames):
            save_path = os.path.join(save_dir, filename)
            save_image(output, save_path)

avg_loss = running_loss / num_samples
avg_ssim = total_ssim / num_samples
print(f"Average Loss: {avg_loss:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")


