import torch
import numpy as np
from utility import *
import time
import struct
import numpy as np
import argparse

import data_loader_r3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors, 6), alpha=.6, **kwargs)

def main(args):
        obc = data_loader_r3d.load_obs_list(args.env_id, folder=args.data_path)
        for path_file in args.path_file:
            # visualize path
            if path_file.endswith('.txt'):
                path = np.loadtxt(path_file)
            else:
                path = np.fromfile(path_file)
            path = path.reshape(-1, 3)
            path_x = []
            path_y = []
            path_z = []
            for i in range(len(path)):
                path_x.append(path[i][0])
                path_y.append(path[i][1])
                path_z.append(path[i][2])



parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../hw3_data/dataset-c3d/')
parser.add_argument('--env-id', type=int, default=0)
parser.add_argument('--point-cloud', default=False, action='store_true')
# parser.add_argument('--path-file', nargs='*', type=str, default=['../hw3_data/dataset-s2d/e0/path1.dat'], help='path file')
parser.add_argument('--path-file', nargs='*', type=str, default=['../hw3_data/dataset-c3d/e0/path3.dat'], help='path file')
args = parser.parse_args()
print(args)
main(args)


def steerTo(start, end, obc, IsInCollision, step_sz=0.01):
    # test if there is a collision free path from start to end, with step size
    # given by step_sz, and with generic collision check function
    # here we assume start and end are tensors
    # return 0 if in coliision; 1 otherwise
    DISCRETIZATION_STEP=step_sz
    delta = end - start  # change
    delta = delta.numpy()
    total_dist = np.linalg.norm(delta)
    # obtain the number of segments (start to end-1)
    # the number of nodes including start and end is actually num_segs+1
    num_segs = int(total_dist / DISCRETIZATION_STEP)
    if num_segs == 0:
        # distance smaller than threshold, just return 1
        return 1
    # obtain the change for each segment
    delta_seg = delta / num_segs
    # initialize segment
    seg = start.numpy()
    # check for each segment, if they are in collision
    for i in range(num_segs+1):
        if IsInCollision(seg, obc):
            # in collision
            return 0
        seg = seg + delta_seg
    return 1