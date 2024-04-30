import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pickle
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
            nn.Linear(16 * (img_width//2) * (img_height//2), 512),
            nn.ReLU(),
            nn.Linear(512, num_joints)
        )

    def to(self, device):
        return super().to(device)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.regressor(x)
        return x

# Function to calculate the accuracy (e.g., RMSE)
def calculate_accuracy(model, data_loader):
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    mean_loss = total_loss / len(data_loader)
    rmse = np.sqrt(mean_loss)
    return rmse

def calculate_percentage_accuracy(model, data_loader, tolerance=0.05):
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            # Calculate the absolute difference and check if within tolerance
            absolute_difference = torch.abs(outputs - labels)
            tolerance_threshold = tolerance * torch.abs(labels)
            # Count predictions within the specified tolerance
            correct_predictions += torch.sum(absolute_difference <= tolerance_threshold)
            total_predictions += labels.numel()

    # Calculate the percentage of predictions within the tolerance
    accuracy = (correct_predictions.float() / total_predictions) * 100
    return accuracy.item() 


##### Load dataset (for limited range of robot configs)
conf_dataset = np.loadtxt('img_dataset/config-test.dat')
with open('img_dataset/rgb-test.dat', 'rb') as file:
    rgb_dataset = pickle.load(file) 

##### Load dataset (for full feasible range of robot configs)
# conf_dataset = np.loadtxt('img_dataset/config-full-test.dat')
# with open('img_dataset/rgb-full-test2.dat', 'rb') as file:
#     rgb_dataset = pickle.load(file)


# Preprocess the RGB dataset
preprocessed_rgb_dataset = preprocess_data(rgb_dataset)
print('loaded and preprocessed dataset')

# Initialize dataset with preprocessed data
dataset = RobotConfigDataset(rgbPixels=preprocessed_rgb_dataset, labels=conf_dataset)
test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

##### Load the trained model (for limited range of robot configs)
model_path = 'models/visuomotor_model.pth'

##### Load the trained model (for full feasible range of robot configs)
# model_path = 'models/visuomotor_model_full.pth'

num_joints = 3
model = VisuomotorCNN(num_joints=3, img_width=200, img_height=200).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f'Testing accuracy for the model: {model_path}')

# Calculate the accuracy of the model on the test dataset
rmse = calculate_accuracy(model, test_loader)
print(f"RMSE of the model on the test dataset: {rmse}")

tolerance=0.05
accuracy = calculate_percentage_accuracy(model, test_loader)
print(f'Accuracy within ±{tolerance*100}% tolerance: {accuracy}%')