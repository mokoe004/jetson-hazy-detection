import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from aodnet import aodnet
from PIL import Image

root_dir = 'path_to_root_directory'

transform = transforms.Compose([
    transforms.Resize((480, 480), interpolation=Image.Resampling.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DehazingDataset(root_dir=root_dir, subset='train', transform=transform)
val_dataset = DehazingDataset(root_dir=root_dir, subset='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = aodnet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['foggy_image'].to(device), data['clear_image'].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*inputs.size(0)

        epoch_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data['foggy_image'].to(device), data['clear_image'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*inputs.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'aodnet.pth')
            
