import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RTTSDataset(Dataset):
    def __init__(self, cfg, image_set='test', transforms=None):
        """
        Args:
            cfg: Das OmegaConf Objekt (aus der yaml geladen)
            image_set: 'train' oder 'test' (entspricht dem Dateinamen in ImageSets/Main)
            transforms: Albumentations oder Torchvision Transformationen
        """
        self.cfg = cfg
        self.transforms = transforms

        self.image_dir = os.path.join(cfg.dataset.root, 'JPEGImages')
        self.annotation_dir = os.path.join(cfg.dataset.root, 'Annotations')

        split_path = os.path.join(cfg.dataset.root, 'ImageSets', 'Main', f'{image_set}.txt')

        with open(split_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]
        #TODO Collect from xml files (i already did this somewhere)
        self.class_to_idx = {
            'person': 1,
            'bicycle': 2,
            'car': 3,
            'motorcycle': 4,
            'bus': 5
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        img_path = os.path.join(self.image_dir, f'{img_id}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f'{img_id}.jpg')

        img = Image.open(img_path).convert("RGB")

        if not self.cfg.dataset.get("return_bboxes", False):
            if self.transforms:
                img = self.transforms(img)
            return img, img_id

        anno_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
        boxes, labels = self._parse_xml(anno_path)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name not in self.class_to_idx:
                continue

            bndbox = obj.find('bndbox')
            coords = [
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)
            ]
            boxes.append(coords)
            labels.append(self.class_to_idx[name])

        return boxes, labels


def rtts_collate_fn(batch):
    return tuple(zip(*batch))

# Dataloader for Training with Hazy/ Clear images
class PairedDataset(Dataset):
    def __init__(self, cfg, transforms=None):
        self.cfg = cfg
        self.hazy_imgs_dir = os.path.join(cfg.dataset.root, 'hazy_images')
        self.clear_imgs_dir = os.path.join(cfg.dataset.root, 'clear_images')
        self.hazy_imgs = sorted(os.listdir(self.hazy_imgs_dir))
        self.clear_imgs = sorted(os.listdir(self.clear_imgs_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.hazy_imgs)

    def __getitem__(self, index):
        hazy_path = os.path.join(self.hazy_imgs_dir, self.hazy_imgs[index])

        # Images are named same in hazy and clear
        clear_name = self.hazy_imgs[index]
        clear_path = os.path.join(self.clear_imgs_dir, clear_name)

        hazy_img = Image.open(hazy_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")

        if self.transforms:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            hazy_img = self.transforms(hazy_img)
            torch.manual_seed(seed)
            clear_img = self.transforms(clear_img)

        return hazy_img, clear_img