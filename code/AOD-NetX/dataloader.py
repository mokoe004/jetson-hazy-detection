import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DehazingDataset(Dataset):
    def __init__(self, clear_dir, foggy_dir, bb_dir, subset='train', transform=None):
        """
        Args:
            original_dir (string): Path to the directory with the original images.
            foggy_dir (string): Path to the directory with the foggy images.
            bb_dir (string): Path to the directory with bounding box annotations.
            subset (string): One of ['train', 'test', 'val'] indicating the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.original_dir = os.path.join(clear_dir, subset)
        self.foggy_dir = os.path.join(foggy_dir, subset)
        self.bb_dir = os.path.join(bb_dir, subset, 'annotations')
        self.transform = transform
        self.images = [f for f in os.listdir(self.foggy_dir) if os.path.isfile(os.path.join(self.foggy_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        clear_image_path = os.path.join(self.original_dir, img_name)
        foggy_image_path = os.path.join(self.foggy_dir, img_name)
        bb_file_path = os.path.join(self.bb_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        clear_image = Image.open(clear_image_path).convert('RGB')
        foggy_image = Image.open(foggy_image_path).convert('RGB')

        bboxes = []
        if os.path.exists(bb_file_path):
            with open(bb_file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        class_label, x_center, y_center, width, height, conf = map(float, parts)
                        bboxes.append([class_label, x_center, y_center, width, height, conf])
        else:
            print(f"Warning: Bounding box file not found for {img_name}")

        sample = {'clear_image': clear_image, 'foggy_image': foggy_image, 'bboxes': bboxes, 'filename': img_name}

        if self.transform:
            sample['clear_image'] = self.transform(sample['clear_image'])
            sample['foggy_image'] = self.transform(sample['foggy_image'])

        return sample