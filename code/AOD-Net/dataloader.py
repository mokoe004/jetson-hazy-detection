import os
from PIL import Image
from torch.utils.data import Dataset

class DehazingDataset(Dataset):
    def __init__(self, root_dir, subset='test', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.foggy_path = os.path.join(root_dir, 'foggy_images', subset)
        self.clear_path = os.path.join(root_dir, 'clear_images', subset)

        self.image_files = [f for f in os.listdir(self.foggy_path) if os.path.isfile(os.path.join(self.foggy_path, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        foggy_image_path = os.path.join(self.foggy_path, filename)
        clear_image_path = os.path.join(self.clear_path, filename)

        #converting the images to RGB
        foggy_image = Image.open(foggy_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        #applying the transform
        if self.transform:
            foggy_image = self.transform(foggy_image)
            clear_image = self.transform(clear_image)

        return {'foggy_image': foggy_image, 'clear_image': clear_image, 'filename': filename}



