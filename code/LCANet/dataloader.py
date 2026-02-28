import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class DehazingDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        self.foggy_path = os.path.join(root_dir, 'foggy_images', subset)
        self.clear_path = os.path.join(root_dir, 'clear_images', subset)

        self.image_files = sorted(os.listdir(self.foggy_path))
        self.image_files = [
            f for f in self.image_files
            if os.path.exists(os.path.join(self.clear_path, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        foggy = Image.open(os.path.join(self.foggy_path, filename)).convert("RGB")
        clear = Image.open(os.path.join(self.clear_path, filename)).convert("RGB")

        # ----- Synchronisierte Augmentation -----
        if self.subset == 'train':

            # Random horizontal flip
            if random.random() > 0.5:
                foggy = TF.hflip(foggy)
                clear = TF.hflip(clear)

            # Random crop
            i, j, h, w = T.RandomCrop.get_params(foggy, output_size=(256, 256))
            foggy = TF.crop(foggy, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)

        # ToTensor + Normalize
        foggy = TF.to_tensor(foggy)
        clear = TF.to_tensor(clear)

        return foggy, clear