import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from aodnetX import DehazeNetAttention
from utils import ssim, calculate_psnr
import numpy as np
from torchvision.utils import save_image

def collate_fn(batch):
    batch_mod = {key: [d[key] for d in batch] for key in batch[0]}
    batch_mod['clear_image'] = torch.stack(batch_mod['clear_image'])
    batch_mod['foggy_image'] = torch.stack(batch_mod['foggy_image'])
    return batch_mod


clear_dir = '../../datasets/hazy-no-bb/GT'
foggy_dir = '../../datasets/hazy-no-bb/hazy'
bb_dir = '../../datasets/hazy-no-bb/bb'

save_dir = './runs'
os.makedirs(save_dir, exist_ok=True) 

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

test_dataset = DehazingDataset(clear_dir, foggy_dir, bb_dir, 'val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.load('./aodnetX.pt').to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Loss Function
loss_fn = nn.MSELoss()
running_loss = 0.0
total_ssim = 0.0
total_psnr = 0.0
num_samples = 0

with torch.no_grad():
    for data in test_loader:
        clear_images = data['clear_image'].to(device)
        foggy_images = data['foggy_image'].to(device)
        filenames = data['filename']

        outputs = model(foggy_images)
        loss = loss_fn(outputs, clear_images)

        outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
        clear_images_np = clear_images.permute(0, 2, 3, 1).cpu().numpy()

        #calculating batch ssim score
        batch_ssim = np.mean([ssim(out, orig) for out, orig in zip(outputs_np, clear_images_np)])

        #calculating PSNR score
        batch_psnr = np.mean([calculate_psnr(out, orig) for out, orig in zip(outputs_np, clear_images_np)])

        running_loss += loss.item() * clear_images.size(0)
        total_ssim += batch_ssim * clear_images.size(0)
        total_psnr += batch_psnr * clear_images.size(0)

        num_samples += clear_images.size(0)

        # Save dehazed images
        for output, filename in zip(outputs, filenames):
            save_path = os.path.join(save_dir, filename)
            save_image(output, save_path)

avg_loss = running_loss / num_samples
avg_ssim = total_ssim / num_samples
avg_psnr = total_psnr/num_samples
print(f"Average Loss: {avg_loss:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average PSNR: {avg_psnr:.4f}")