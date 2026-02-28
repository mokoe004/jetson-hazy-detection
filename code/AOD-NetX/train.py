import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from aodnetX import DehazeNetAttention

from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

def collate_fn(batch):
    """ Custom collate function to handle batches with variable-sized bounding boxes. """
    batch_mod = {key: [d[key] for d in batch] for key in batch[0]}
    batch_mod['clear_image'] = default_collate(batch_mod['clear_image'])
    batch_mod['foggy_image'] = default_collate(batch_mod['foggy_image'])
    # Do not attempt to collate 'bboxes' as they are lists of varying sizes
    return batch_mod
# Set the directories for your dataset


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        clear_images = data['clear_image'].to(device)
        foggy_images = data['foggy_image'].to(device)
        bounding_boxes = data['bboxes']

        optimizer.zero_grad()

        # Forward pass
        outputs = model(foggy_images, bounding_boxes)
        loss = loss_fn(outputs, clear_images)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clear_images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            clear_images = data['clear_image'].to(device)
            foggy_images = data['foggy_image'].to(device)
            bounding_boxes = data['bboxes']

            outputs = model(foggy_images, bounding_boxes)
            loss = loss_fn(outputs, clear_images)

            running_loss += loss.item() * clear_images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Dataset paths and transformation
    clear_dir = '../../datasets/hazy-no-bb/GT'
    foggy_dir = '../../datasets/hazy-no-bb/hazy'
    bb_dir = '../../datasets/hazy-no-bb/bb'
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    # Datasets and Dataloaders
    train_dataset = DehazingDataset(clear_dir, foggy_dir, bb_dir, 'train', transform=transform)
    val_dataset = DehazingDataset(clear_dir, foggy_dir, bb_dir, 'val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model
    model = DehazeNetAttention().to(device)

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()

    torch.save(model, './aodnetX.pt')

if __name__ == "__main__":
    main()
