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
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
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
    n = math.floor(L/.05)
    if n > 0:
        dir = .05*diff/L
    colide = collision_fn(rand_conf)
    i = 0
    while i < n:
        if colide == True: break
        i+=1
        new_node = i*dir+nearest_conf
        colide = collision_fn(new_node)
    return colide

def prm_planning(start_conf, goal_conf):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    sample_x, sample_y, sample_z = sample_points(start_conf, goal_conf)

    road_map = generate_road_map(sample_x, sample_y, sample_z)

    rx, ry, rz = dijkstra_planning(start_conf, goal_conf, road_map, sample_x, sample_y, sample_z)

    path_conf = []
    for i in reversed(range(len(rx))):
        path_conf.append([rx[i], ry[i], rz[i]])
    return path_conf

def dijkstra_planning(start_conf, goal_conf, road_map, sample_x, sample_y, sample_z):
    open_set, closed_set = dict(), dict()
    start_node = Node(start_conf, 0, -1)
    goal_node = Node(goal_conf, 0, -1)
    open_set[len(road_map) - 2] = start_node
    path_found = True
    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if c_id == (len(road_map) - 1):
            print("goal is found!")
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
            d = math.hypot(dx, dy, dz)
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



def sample_points(start_conf, goal_conf):
    sample_x, sample_y, sample_z = [], [], []
    while len(sample_x) <= N_SAMPLE:
        # tx = np.random.uniform(-2*np.pi, 2*np.pi)
        # ty = np.random.uniform(-2*np.pi, 2*np.pi)
        # tz = np.random.uniform(-np.pi, np.pi)
        tx = np.random.uniform(goal_conf[0]-1.6, goal_conf[0]+.1)
        ty = np.random.uniform(goal_conf[1]-.1, goal_conf[1]+.5)
        tz = np.random.uniform(goal_conf[2]-.1, goal_conf[2]+.5)
        rand_conf = [tx, tz, ty]
        colide = collision_fn(rand_conf)
        if not colide:
            sample_x.append(tx)
            sample_y.append(ty)
            sample_z.append(tz)

    sample_x.append(start_conf[0])
    sample_y.append(start_conf[1])
    sample_z.append(start_conf[2])
    sample_x.append(goal_conf[0])
    sample_y.append(goal_conf[1])
    sample_z.append(goal_conf[2])
    return sample_x, sample_y, sample_z

def generate_road_map(sample_x, sample_y, sample_z):
    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y, sample_z)).T)

    for (i, ix, iy, iz) in zip(range(n_sample), sample_x, sample_y, sample_z):
        dists, indexes = sample_kd_tree.query([ix, iy, iz], k=n_sample)
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

###############################################################################
#your implementation ends here

if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    path_conf = prm_planning(start_conf, goal_conf)


    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.3)
            time.sleep(1)