from __future__ import print_function
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
from Model.end2end_model import End2EndMPNet
import Model.model as model
import Model.AE.CAE_3d as CAE_3d
import numpy as np
import argparse
from plan_general import *
import data_loader_r3d
import os
import random
from utility import *
import utility_c3d
import progressbar
import pybullet as p
import pybullet_data

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
        # rand_conf1 = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]
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
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=27.5, cameraYaw=90.000, cameraPitch=-60.00,
                                 cameraTargetPosition=(-1.5, 7.5, 1.5))

    # load objects
    plane = p.loadURDF("plane.urdf")
    obstacles = [plane]
    obstacle_xyz = [[10.374696, 12.402063, 10.9207115],
                    [-4.187111, 9.439998, -14.205707],
                    [12.065512, 4.9649725, -10.829719],
                    [4.6687045, 4.034849, -0.5135859],
                    [-0.71889645, -0.38771427, 5.7259583],
                    [-11.277511, 3.5690885, -14.776423],
                    [-9.015153, -1.0562156, 12.62564],
                    [-12.584196, 1.1506163, 9.944185],
                    [2.3307815, -9.180679, -1.0209659],
                    [-5.455794, -6.490853, 7.518958]
                    ]

    for i, obs in enumerate(obstacle_xyz):
        obstacle = p.loadURDF(f'assets/blocks/block{i + 1}.urdf',
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

    conf_dataset = [[-.6, -.5, -.75], [.7, -.5, -.75]]
    rgb_dataset = []

    for conf in conf_dataset:
        set_joint_positions(ur5, UR5_JOINT_INDICES, conf)
        p.stepSimulation()
        rgb_img = p.getCameraImage(200, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        rgb_dataset.append(rgb_img)

    # Preprocess the RGB dataset
    preprocessed_rgb_dataset = preprocess_data(rgb_dataset)
    print('loaded and preprocessed dataset')

    # Initialize dataset with preprocessed data
    dataset = RobotConfigDataset(rgbPixels=preprocessed_rgb_dataset, labels=conf_dataset)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Load the trained model
    num_joints = 3
    model_visual = VisuomotorCNN(num_joints=3, img_width=200, img_height=200).to(device)
    model_visual.load_state_dict(torch.load('models/visuomotor_model.pth'))
    model_visual.eval()

    # Compare the model output with true robot config
    for inputs, labels in test_loader:
        outputs = model_visual(inputs)

    # Start MP Net
    total_input_size = 6000+6
    AE_input_size = 6000
    mlp_input_size = 28+6
    output_size = 3

    IsInCollision = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                    attachments=[], self_collisions=True,
                                    disabled_collisions=set())

    load_test_dataset = data_loader_r3d.load_test_dataset
    normalize = utility_c3d.normalize
    unnormalize = utility_c3d.unnormalize
    CAE = CAE_3d
    MLP = model.MLP
    mpNet = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size, \
                output_size, CAE, MLP)

    model_path='mpnet_epoch_15000.pkl'
    if True:
        load_net_state(mpNet, 'models/mpnet_epoch_15000.pkl')
    test_data = load_test_dataset(N=1, NP=1, s=2, sp=1990, folder='../milestone2/')
    obc, obs, _, _ = test_data
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()

    normalize_func=lambda x: normalize(x, np.pi)
    unnormalize_func=lambda x: unnormalize(x, np.pi)

    # test on dataset
    test_suc_rate = 0.
    DEFAULT_STEP = 0.01
    # for statistics

    n_valid_total = 0
    n_successful_total = 0
    sum_time = 0.0
    sum_timesq = 0.0
    min_time = float('inf')
    max_time = -float('inf')

    for i in range(1):
        start_conf = outputs[0].cpu().detach().numpy()
        end_conf = outputs[1].cpu().detach().numpy()
        paths = np.array([start_conf, end_conf])
        n_valid_cur = 0
        n_successful_cur = 0

        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in range(1):
            time0 = time.time()
            found_path = False
            n_valid_cur += 1
            path = [torch.from_numpy(paths[0]).type(torch.FloatTensor),\
                    torch.from_numpy(paths[1]).type(torch.FloatTensor)]
            step_sz = DEFAULT_STEP
            MAX_NEURAL_REPLAN = 20
            for t in range(MAX_NEURAL_REPLAN):
                path = neural_plan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                    normalize_func, unnormalize_func, t==0, step_sz=step_sz)
                path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                if feasibility_check(path, obc[i], IsInCollision, step_sz=step_sz):
                    found_path = True
                    n_successful_cur += 1
                    break

            time1 = time.time() - time0
            sum_time += time1
            sum_timesq += time1 * time1
            min_time = min(min_time, time1)
            max_time = max(max_time, time1)

            # write the path
            if type(path[0]) is not np.ndarray:
                # it is torch tensor, convert to numpy
                path = [p.numpy() for p in path]
            path = np.array(path)
            path_file = '../milestone2/result/env_2/'

            if found_path:
                filename = f'path_test.txt'
            else:
                filename = f'path_test-fail.txt'
            np.savetxt(path_file + filename, path, fmt='%f')

        set_joint_positions(ur5, UR5_JOINT_INDICES, path[0])
        time.sleep(1)
        for q in path:
            set_joint_positions(ur5, UR5_JOINT_INDICES, q)
            time.sleep(1)