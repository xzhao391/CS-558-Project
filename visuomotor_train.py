import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pickle
import numpy as np
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocess dataset
def preprocess_data(rgbPixels, height=200, width=200):
    rgb_tensor = torch.tensor(rgbPixels, dtype=torch.float).to(device)  # Convert to tensor and move to device
    rgb_tensor = rgb_tensor.view(-1, height, width, 4)  # Assuming -1 for batch size, and 4 for RGBA channels
    rgb_tensor = rgb_tensor[:, :, :, :3]  # Drop the alpha channel to have only RGB
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, height, width]

    # Normalize the RGB values to [0, 1]
    rgb_tensor /= 255.0

    return rgb_tensor

# Dataset class
class RobotConfigDataset(Dataset):
    def __init__(self, rgbPixels, labels):
        # self.rgbPixels = torch.tensor(rgbPixels, dtype=torch.float).to(device)
        self.rgbPixels = rgbPixels
        self.labels = torch.tensor(labels, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb = self.rgbPixels[idx]
        label = self.labels[idx]
        return rgb, label

# CNN model
class VisuomotorCNN(nn.Module):
    def __init__(self, img_width, img_height, num_joints):
        super(VisuomotorCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(16 * (img_width//2) * (img_height//2), 512),  # Adjust size according to your input image size and conv layers
            nn.ReLU(),
            nn.Linear(512, num_joints)  # num_joints is the number of robot joint configurations you have
        )

    def to(self, device):
        return super().to(device)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.regressor(x)
        return x

# Load and preprocess dataset
conf_dataset = np.loadtxt('img_dataset/config.dat')
with open('img_dataset/rgb.dat', 'rb') as file:
    rgb_dataset = pickle.load(file) 

# Preprocess the RGB dataset using preprocess_data()
preprocessed_rgb_dataset = preprocess_data(rgb_dataset)
print('loaded and preprocessed dataset')

# Initialize dataset with preprocessed data
dataset = RobotConfigDataset(rgbPixels=preprocessed_rgb_dataset, labels=conf_dataset)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = VisuomotorCNN(num_joints=3, img_width=200, img_height=200).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
print('begin training..')
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
model_path = 'models/visuomotor_model.pth'
torch.save(model.state_dict(), model_path)
print('saved at ' + model_path)