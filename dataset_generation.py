

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
import csv
# PRMstar
from scipy.spatial import KDTree

UR5_JOINT_INDICES = [0, 1, 2]

# parameter
N_SAMPLE = 500  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length


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


class Node:
    def __init__(self, conf, cost, parent_index):
        self.conf = conf
        self.cost = cost
        self.parent_index = parent_index


def steer_to(rand_conf, nearest_conf):
    rand_conf = np.array(rand_conf)
    nearest_conf = np.array(nearest_conf)
    diff = rand_conf - nearest_conf
    L = np.linalg.norm(diff)
    n = math.floor(L / .05)
    if n > 0:
        dir = .05 * diff / L
    colide = collision_fn(rand_conf)
    i = 0
    while i < n:
        if colide == True: break
        i += 1
        new_node = i * dir + nearest_conf
        colide = collision_fn(new_node)
    return colide


def prm_planning(road_map, start_list, goal_list):
    sample_x, sample_y, sample_z = sample_points(start_list, goal_list)

    if road_map is None:
        road_map = generate_road_map(sample_x, sample_y, sample_z)

    path_list = []
    n = len(start_list)
    for i in range(n):
        rx, ry, rz = dijkstra_planning(start_list[i], goal_list[i], n - i - 1, road_map, sample_x, sample_y, sample_z)

        path_conf = []
        for j in reversed(range(len(rx))):
            path_conf.append([rx[j], ry[j], rz[j]])
        path_conf = smoothing(path_conf)
        path_list.append(path_conf)
        print(i)
    return path_list

def smoothing(path):
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    if len(path) <= 2:
        return path
    else:
        for i in range(100):
            n = len(path)-1
            index1 = random.randint(0, n)
            index2 = random.randint(0, n)
            colide = steer_to(path[index1], path[index2])
            if not colide:
                path = np.delete(path, slice(min(index1,index2)+1,max(index1,index2)), 0)
    path = path.tolist()
    return path

def dijkstra_planning(start_conf, goal_conf, reverse_i, road_map, sample_x, sample_y, sample_z):
    open_set, closed_set = dict(), dict()
    start_node = Node(start_conf, 0, -1)
    goal_node = Node(goal_conf, 0, -1)
    open_set[len(road_map) - 2 * reverse_i - 2] = start_node
    path_found = True
    while True:
        if not open_set:
            # print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if c_id == (len(road_map) - 2 * reverse_i - 1):
            # print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.conf[0]
            dy = sample_y[n_id] - current.conf[1]
            dz = sample_z[n_id] - current.conf[2]
            d = np.linalg.norm([dx, dy, dz])
            node = Node([sample_x[n_id], sample_y[n_id], sample_z[n_id]]
                        , current.cost + d, c_id)
            if n_id in closed_set:
                continue
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], [], []
    rx, ry, rz = [goal_node.conf[0]], [goal_node.conf[1]], [goal_node.conf[2]]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.conf[0])
        ry.append(n.conf[1])
        rz.append(n.conf[2])
        parent_index = n.parent_index
    return rx, ry, rz


def sample_points(start_list, goal_list):
    sample_x, sample_y, sample_z = [], [], []
    while len(sample_x) <= N_SAMPLE:
        tx = np.random.uniform(-2 * np.pi, 2 * np.pi)
        ty = np.random.uniform(-2 * np.pi, 2 * np.pi)
        tz = np.random.uniform(-np.pi, np.pi)
        rand_conf = [tx, tz, ty]
        colide = collision_fn(rand_conf)
        if not colide:
            sample_x.append(tx)
            sample_y.append(ty)
            sample_z.append(tz)

    for i in range(len(start_list)):
        sample_x.append(start_list[i][0])
        sample_y.append(start_list[i][1])
        sample_z.append(start_list[i][2])
        sample_x.append(goal_list[i][0])
        sample_y.append(goal_list[i][1])
        sample_z.append(goal_list[i][2])
    return sample_x, sample_y, sample_z


def generate_road_map(sample_x, sample_y, sample_z):
    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y, sample_z)).T)
    d = 3  # dimension of the space
    gamma_prm = 50

    for (iter, ix, iy, iz) in zip(range(n_sample), sample_x, sample_y, sample_z):
        k = 0
        if iter is not 0:
            k = int(gamma_prm * (np.log(iter) / iter) ** (1.0 / d))
        k = max(1, min(k, iter - 1))
        # print(f'iter={iter}, k={k}')

        dists, indexes = sample_kd_tree.query([ix, iy, iz], k=k + 1)
        edge_id = []
        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]
            nz = sample_z[indexes[ii]]

            colide = steer_to([ix, iy, iz], [nx, ny, nz])
            if not colide:
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break
        road_map.append(edge_id)
    return road_map


if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=35.0, cameraYaw=158.000, cameraPitch=-40.00,
                                 cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    obstacles = []
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
                    [ -5.455794  ,  -6.490853  ,   7.518958  ]]
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

    # initialize road map
    road_map = None
    start_list = []
    goal_list = []
    i = 1999
    while i < 2000:
        # start_node = (np.random.uniform(-np.pi, np.pi),
        #               np.random.uniform(-np.pi, np.pi),
        #               np.random.uniform(-np.pi, np.pi))
        # goal_node = (np.random.uniform(-np.pi, np.pi),
        #              np.random.uniform(-np.pi, np.pi),
        #              np.random.uniform(-np.pi, np.pi))
        # start_node = [-0.6+np.random.uniform(-1, 1), -0.5+np.random.uniform(-1, 1), -0.75+np.random.uniform(-1, 1)]
        # goal_node = [.7+np.random.uniform(-1, 1), -0.5+np.random.uniform(-1, 1), -0.75+np.random.uniform(-1, 1)]
        start_node = [-0.6, -0.5, -0.75]
        goal_node = [.7, -0.5, -0.75]
        if (not collision_fn(start_node)) and (not collision_fn(goal_node)):
            start_list.append(start_node)
            goal_list.append(goal_node)
            i += 1
    path_list = prm_planning(road_map, start_list, goal_list)
    path = 2000
    folder = '../milestone2/path/path'
    for i in range(len(start_list)):
        if path_list[i] == []:
            continue
        else:
            np.savetxt(folder+str(path)+'.dat', path_list[i])
            path += 1
            # for q in path_list[i]:
            #     set_joint_positions(ur5, UR5_JOINT_INDICES, q)
            #     time.sleep(2)

# start_node = [-0.6, -0.5, -0.75]
# set_joint_positions(ur5, UR5_JOINT_INDICES, start_node)
# collision_fn(start_node)
#
# goal_node = [.7, -0.5, -0.75]
# set_joint_positions(ur5, UR5_JOINT_INDICES, goal_node)
# collision_fn(goal_node)