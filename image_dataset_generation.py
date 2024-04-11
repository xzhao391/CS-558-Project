from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import sys
import random
from scipy.spatial.transform import Rotation as R
import csv
import pickle

# PRMstar
from scipy.spatial import KDTree


UR5_JOINT_INDICES = [0, 1, 2]

# parameter
N_SAMPLE = 200  # number of sample_points

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)

def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id

def remove_marker(marker_id):
   p.removeBody(marker_id)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=500, help='number of sample points')
    parser.add_argument('--n-knn', type=int, default=10, help='number of sample points')
    parser.add_argument('--max-edge-len', type=float, default=30.0, help='number of sample points')
    args = parser.parse_args()
    return args

def sample_robot_configs():
    configs = []
    while len(configs) <= N_SAMPLE:
        # tx = np.random.uniform(-2*np.pi, 2*np.pi)
        # ty = np.random.uniform(-2*np.pi, 2*np.pi)
        # tz = np.random.uniform(-np.pi, np.pi)
        rand_conf1 = [-0.6+np.random.uniform(-1, 1), -0.5+np.random.uniform(-1, 1), -0.75+np.random.uniform(-1, 1)]
        rand_conf2 = [.7+np.random.uniform(-1, 1), -0.5+np.random.uniform(-1, 1), -0.75+np.random.uniform(-1, 1)]
        # rand_conf = [tx, tz, ty]
        # colide = collision_fn(rand_conf)
        if not collision_fn(rand_conf1):
            configs.append(rand_conf1)
        if not collision_fn(rand_conf2):
            configs.append(rand_conf2)

    return configs

def draw_frame(position, quaternion=[0, 0, 0, 1]):
    m = R.from_quat(quaternion).as_matrix()
    x_vec = m[:, 0]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        p.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

def get_ur5_camera_transform(ur5):
    conf = (0.2, -1.5, 1.5)
    set_joint_positions(ur5, UR5_JOINT_INDICES, conf)
    p.stepSimulation()

    world_to_eef = p.getLinkState(ur5, 3, computeForwardKinematics=True)[:2]  # eef pose
    xA, yA, zA = world_to_eef[0]
    zA += 1.0

    xB = xA + 1.0
    yB = yA 
    zB = zA

    world_to_cameraEyePose = ((xA, yA, zA), world_to_eef[1])
    world_to_cameraTargetPose = ((xB, yB, zB), world_to_eef[1])
    cameraEyePose_to_world = p.invertTransform(world_to_cameraEyePose[0], world_to_cameraEyePose[1])
    eye_to_target = p.multiplyTransforms(cameraEyePose_to_world[0], cameraEyePose_to_world[1],
                                         world_to_cameraTargetPose[0], world_to_cameraTargetPose[1])
    
    return eye_to_target

def ur5_mounted_camera(ur5, camera_eye_to_target):
    img_w, img_h = 100, 100

    world_to_eef = p.getLinkState(ur5, 3, computeForwardKinematics=True)[:2]  # eef pose
    xA, yA, zA = world_to_eef[0]
    zA = zA + 1.0  # make the camera a little higher than the robot

    world_to_cameraEyePose = ((xA, yA, zA), world_to_eef[1])

    world_to_cameraTargetPose = p.multiplyTransforms(world_to_cameraEyePose[0], world_to_cameraEyePose[1],
                                                     camera_eye_to_target[0], camera_eye_to_target[1])

    # draw_frame(world_to_cameraEyePose[0], world_to_cameraEyePose[1])
    # draw_frame(world_to_cameraTargetPose[0], world_to_cameraTargetPose[1])

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=world_to_cameraEyePose[0],
                        cameraTargetPosition=world_to_cameraTargetPose[0],
                        cameraUpVector=[0, 0, 1.0]
                    )

    projection_matrix = p.computeProjectionMatrixFOV(
                            fov=90, aspect=1.5, nearVal=0.02, farVal=10)

    imgs = p.getCameraImage(img_w, img_h,
                            view_matrix,
                            projection_matrix, shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return imgs

if __name__ == "__main__":
    args = get_args()

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

    conf_list = sample_robot_configs()
    rgb_img_list = []

    for conf in conf_list:
        set_joint_positions(ur5, UR5_JOINT_INDICES, conf)
        p.stepSimulation()
        rgb_img = p.getCameraImage(200, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        rgb_img_list.append(rgb_img)
    
    np.savetxt('config-test.dat', conf_list)

    with open('rgb-test.dat', 'wb') as file:
        pickle.dump(np.array(rgb_img_list), file)

    with open('rgb-test.dat', 'rb') as file:
        loaded_data = pickle.load(file)
    print(loaded_data.shape)
    print(loaded_data[0])