import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# test file for RESIDE- RTTS dataset
class DehazingDataset(Dataset):
    def __init__(self, foggy_dir, bb_dir, transform=None):
        """
        Args:
            foggy_dir (string): Path to the directory with the foggy images.
            bb_dir (string): Path to the directory with bounding box annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.foggy_dir = foggy_dir
        self.bb_dir = os.path.join(bb_dir, 'Annotations')
        self.transform = transform
        self.images = [f for f in os.listdir(self.foggy_dir) if os.path.isfile(os.path.join(self.foggy_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        foggy_image_path = os.path.join(self.foggy_dir, img_name)
        bb_file_path = os.path.join(self.bb_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        foggy_image = Image.open(foggy_image_path).convert('RGB')

        bboxes = []
        if os.path.exists(bb_file_path):
            with open(bb_file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        class_label, x_center, y_center, width, height, conf = map(float, parts)
                        bboxes.append([class_label, x_center, y_center, width, height, conf])

        sample = {'foggy_image': foggy_image, 'bboxes': bboxes, 'filename': img_name}

        if self.transform:
            sample['foggy_image'] = self.transform(sample['foggy_image'])

        return sample


import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from aodnet import DehazeNetAttention
from torchvision.utils import save_image

foggy_dir = 'path_to_foggy_images_directory'
bb_dir = 'path_to_bounding_boxes_directory'
save_dir = '/path_to_save_dehazed_images_directory'
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

test_dataset = DehazingDataset(foggy_dir=foggy_dir, bb_dir=bb_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.load('path/aodnetX.pt').to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

with torch.no_grad():
    for data in test_loader:
        foggy_images = data['foggy_image'].to(device)
        filenames = data['filename']  # Extract filenames from the dataset
        bounding_boxes = data['bboxes']

        outputs = model(foggy_images, bounding_boxes)  # Assuming the model takes only the images

        # Save each output image using the original filename
        for output, filename in zip(outputs, filenames):
            save_path = os.path.join(save_dir, filename)
            save_image(output, save_path)
