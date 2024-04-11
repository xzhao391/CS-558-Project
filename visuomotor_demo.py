import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pybullet as p
import pybullet_data
import numpy as np
import time
import pickle
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

UR5_JOINT_INDICES = [0, 1, 2]
N_SAMPLE = 4

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)

def sample_robot_configs():
    configs = []
    while len(configs) <= N_SAMPLE:
        # tx = np.random.uniform(-2*np.pi, 2*np.pi)
        # ty = np.random.uniform(-2*np.pi, 2*np.pi)
        # tz = np.random.uniform(-np.pi, np.pi)
        rand_conf1 = [-0.6+np.random.uniform(-1, 1), -0.5+np.random.uniform(-1, 1), -0.75+np.random.uniform(-1, 1)]
        # rand_conf2 = [.7+np.random.uniform(-1, 1), -0.5+np.random.uniform(-1, 1), -0.75+np.random.uniform(-1, 1)]
        # rand_conf = [tx, tz, ty]
        # colide = collision_fn(rand_conf)
        if not collision_fn(rand_conf1):
            configs.append(rand_conf1)
            break
        # if not collision_fn(rand_conf2):
        #     configs.append(rand_conf2)

    return configs

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

if __name__ == "__main__":
    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=27.5, cameraYaw=90.000, cameraPitch=-60.00, cameraTargetPosition=(-1.5, 7.5, 1.5))

    # load objects
    plane = p.loadURDF("plane.urdf")
    obstacles = [plane]
    obstacle_xyz = [[ 10.374696 ,   12.402063  ,  10.9207115 ],
                    [ -4.187111 ,    9.439998  , -14.205707  ],
                    [ 12.065512 ,    4.9649725 , -10.829719  ],
                    [  4.6687045,    4.034849  ,  -0.5135859 ],
                    [ -0.71889645,  -0.38771427,   5.7259583 ],
                    [-11.277511  ,   3.5690885 , -14.776423  ],
                    [ -9.015153  ,  -1.0562156 ,  12.62564   ],
                    [-12.584196  ,   1.1506163 ,   9.944185  ],
                    [  2.3307815 ,  -9.180679  ,  -1.0209659 ],
                    [ -5.455794  ,  -6.490853  ,   7.518958  ]
                    ]

    for i, obs in enumerate(obstacle_xyz):
        obstacle = p.loadURDF(f'assets/blocks/block{i+1}.urdf',
                            basePosition=obs,
                            useFixedBase=True)
        obstacles.append(obstacle)

    # load robot
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 10, 0.2], useFixedBase=True, globalScaling=20)

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    # Get dataset
    conf_dataset = sample_robot_configs()
    rgb_dataset = []

    for conf in conf_dataset:
        set_joint_positions(ur5, UR5_JOINT_INDICES, conf)
        p.stepSimulation()
        rgb_img = p.getCameraImage(200, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        rgb_dataset.append(rgb_img)
        # time.sleep(10)

    # Preprocess the RGB dataset
    preprocessed_rgb_dataset = preprocess_data(rgb_dataset)
    print('loaded and preprocessed dataset')

    # Initialize dataset with preprocessed data
    dataset = RobotConfigDataset(rgbPixels=preprocessed_rgb_dataset, labels=conf_dataset)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Load the trained model
    num_joints = 3
    model = VisuomotorCNN(num_joints=3, img_width=200, img_height=200).to(device)
    model.load_state_dict(torch.load('models/visuomotor_model.pth'))
    model.eval()

    # Compare the model output with true robot config
    for inputs, labels in test_loader:
        outputs = model(inputs)
        print(f'model prediction: {outputs}')
        print(f'true robot config: {labels}')

