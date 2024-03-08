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

UR5_JOINT_INDICES = [0, 1, 2]


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
class RRT_Node:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.child = child

def sample_conf():
    q_rand = RRT_Node([np.random.uniform(-2*np.pi, 2*np.pi),
                       np.random.uniform(-2*np.pi, 2*np.pi),
                       np.random.uniform(-np.pi, np.pi)])
    # q_rand = RRT_Node([np.random.uniform(goal_conf[0]-1.6, goal_conf[0]+.1),
    #                    np.random.uniform(goal_conf[1]-.1, goal_conf[1]+.5),
    #                    np.random.uniform(goal_conf[2]-.1, goal_conf[2]+.5)])
    flag = np.all(np.isclose(goal_conf, q_rand.conf, rtol=.05))
    return q_rand, flag

def find_nearest(rand_node, node_list):
    temp_dist = []
    for i in range(len(node_list)):
        node = node_list[i]
        dist = np.linalg.norm(rand_node.conf-node.conf)
        temp_dist.append(dist)
    return node_list[temp_dist.index(min(temp_dist))]
        
def steer_to(rand_node, nearest_node):
    diff = rand_node.conf - nearest_node.conf
    L = np.linalg.norm(diff)
    n = math.floor(L/.05)
    if n > 0:
        dir = .05*diff/L
    colide = collision_fn(rand_node.conf)
    i = 0
    while i < n:
        if colide == True: break
        i+=1
        new_node = i*dir+nearest_node.conf
        colide = collision_fn(new_node)
    return colide


def steer_to_until(rand_node, nearest_node):
    diff = rand_node.conf - nearest_node.conf
    L = np.linalg.norm(diff)
    dir = .05*diff/L
    n = math.floor(L / .05)
    colide = False
    i = 0
    while i < n:
        if colide == True: break
        i+=1
        new_node = i*dir+nearest_node.conf
        colide = collision_fn(new_node)
    new_node = RRT_Node((i-1)*dir+nearest_node.conf)
    return new_node

def RRT():
    ###############################################
    # TODO your code to implement the rrt algorithm
    ###############################################
    T = [RRT_Node(start_conf)]
    pathFound = False
    # for i in range(10**5):
    while pathFound == False:
        q_rand, flag = sample_conf()
        q_near = find_nearest(q_rand, T)
        colide = steer_to(q_rand, q_near)

        if not colide:
            q_rand.set_parent(q_near)
            T.append(q_rand)
            if flag:
                pathFound = True
    return Gen_path(q_rand)

def BiRRT():
    #################################################
    # TODO your code to implement the birrt algorithm
    #################################################
    T_start = [RRT_Node(start_conf)]
    T_goal = [RRT_Node(goal_conf)]
    pathFound = False
    i = 0
    while pathFound == False:
        i += 1
        q_rand, flag = sample_conf()
        if i % 2 == 0:
            Ta = T_start
            Tb = T_goal
        else:
            Ta = T_goal
            Tb = T_start

        q_neara = find_nearest(q_rand, Ta)
        q_steer= steer_to_until(q_rand, q_neara)
        q_steer.set_parent(q_neara)
        if i % 2 == 0:
            T_start.append(q_steer)
        else:
            T_goal.append(q_steer)
        # Tb
        q_nearb = find_nearest(q_steer, Tb)
        colide = steer_to(q_rand, q_nearb)
        if not colide:
            pathFound = True

    first_path = Gen_path(q_steer)
    second_path = Gen_path(q_nearb)
    if i % 2 == 0:
        path = np.concatenate((first_path,np.flip(second_path,axis=0)), axis=0)
    else:
        path = np.concatenate((second_path, np.flip(first_path, axis=0)), axis=0)
    return path

def Gen_path(node):
    last_node = node
    path = np.array(last_node.conf).reshape(1,3)
    while last_node.parent:
        last_node = last_node.parent
        path = np.concatenate((last_node.conf.reshape(1,3),path), axis=0)
    return path

def BiRRT_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    path = BiRRT()
    print(path.shape[0])
    for i in range(200):
        n = path.shape[0]-1
        index1 = random.randint(0, n)
        index2 = random.randint(0, n)
        colide = steer_to(RRT_Node(path[index1]), RRT_Node(path[index2]))
        if not colide:
            path = np.delete(path, slice(min(index1,index2)+1,max(index1,index2)), 0)
    return path
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

    
		# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            path_conf = BiRRT_smoothing()
        else:
            # using birrt without smoothing
            path_conf = BiRRT()
    else:
        # using rrt
        path_conf = RRT()
        # path_conf = BiRRT()
        # path_conf = BiRRT_smoothing()

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