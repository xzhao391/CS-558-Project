from __future__ import print_function
from Model.end2end_model import End2EndMPNet
import Model.model as model
import Model.AE.CAE as CAE_2d
import Model.AE.CAE_3d as CAE_3d
import numpy as np
import argparse
import os
import torch
from plan_general import *
import plan_c3d
import data_loader_2d
import data_loader_r3d
from torch.autograd import Variable
import copy
import os
import random
from utility import *
import utility_c3d
import progressbar
import pybullet as p
import pybullet_data

def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    total_input_size = 6000+6
    AE_input_size = 6000
    mlp_input_size = 28+6
    output_size = 3

    # set up simulator
    UR5_JOINT_INDICES = [0, 1, 2]

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
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # load previously trained model if start epoch > 0
    model_path='mpnet_epoch_%d.pkl' %(args.epoch)
    if args.epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        if args.reproducible:
            # set seed from model file
            torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
            torch.manual_seed(torch_seed)
            np.random.seed(np_seed)
            random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
    if args.epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))


    # load test data
    print('loading...')
    test_data = load_test_dataset(N=args.N, NP=args.NP, s=args.s, sp=args.sp, folder=args.data_path)
    obc, obs, _, _ = test_data

    normalize_func=lambda x: normalize(x, args.world_size)
    unnormalize_func=lambda x: unnormalize(x, args.world_size)

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
        start_conf = np.array([0.239459, -0.911111, -1.468722])
        end_conf = np.array([0.842212, 0.057618, -0.473864])
        paths = np.array([start_conf, end_conf])
        n_valid_cur = 0
        n_successful_cur = 0

        widgets = [
            f'planning: env={args.s + i}, path=',
            progressbar.Variable('path_number', format='{formatted_value}', width=1), ' ',
            progressbar.Bar(),
            ' (', progressbar.Percentage(), ' complete)',
            ' success rate = ', progressbar.Variable('success_rate', format='{formatted_value}', width=4, precision=3),
            ' planning time = ', progressbar.Variable('planning_time', format='{formatted_value}sec', width=4, precision=3),
        ]
        bar = progressbar.ProgressBar(widgets = widgets)

        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in bar(range(1)):
            time0 = time.time()
            found_path = False
            n_valid_cur += 1
            path = [torch.from_numpy(paths[0]).type(torch.FloatTensor),\
                    torch.from_numpy(paths[1]).type(torch.FloatTensor)]
            step_sz = DEFAULT_STEP
            MAX_NEURAL_REPLAN = 11
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
            path_file = args.result_path+'env_%d/' % (i+args.s)
            if not os.path.exists(path_file):
                # create directory if not exist
                os.makedirs(path_file)

            if found_path:
                filename = f'path_{j+args.sp}.txt'
            else:
                filename = f'path_{j+args.sp}-fail.txt'
            np.savetxt(path_file + filename, path, fmt='%f')

            success_rate = n_successful_cur / n_valid_cur if n_valid_cur > 0 else float('nan')

            if found_path:
                bar.update(path_number=j+args.sp, success_rate=success_rate, planning_time=time1)

        time.sleep(2)
        for q in path:
            set_joint_positions(ur5, UR5_JOINT_INDICES, q)
            time.sleep(2)
        n_valid_total += n_valid_cur
        n_successful_total += n_successful_cur
        if n_valid_total == 0:
            success_rate = avg_time = stdev_time = float('nan')
        else:
            success_rate = n_successful_total / n_valid_total if n_valid_total > 0 else float('nan')
            avg_time = sum_time / n_valid_total
            stdev_time = np.sqrt((sum_timesq - sum_time * avg_time) / (n_valid_total - 1)) if n_valid_total > 1 else 0
        print(f'valid num={n_valid_total:.2f}, cumulative: success rate={success_rate:.2f}, runtime (min/avg/max/stdev) = {min_time:.2f}/{avg_time:.2f}/{max_time:.2f}/{stdev_time:.2f}s')
def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./models/',help='folder of trained model')
    parser.add_argument('--N', type=int, default=1, help='number of environments')
    parser.add_argument('--NP', type=int, default=1, help='number of paths per environment')
    parser.add_argument('--s', type=int, default=2, help='start of environment index')
    parser.add_argument('--sp', type=int, default=1951, help='start of path index')

    # Model parameters
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data-path', type=str, default='../milestone2/', help='path to dataset')
    parser.add_argument('--result-path', type=str, default='../milestone2/result/', help='folder to save paths computed')
    parser.add_argument('--epoch', type=int, default=10000, help='epoch of trained model to use')
    parser.add_argument('--env-type', type=str, default='c2d', help='s2d for simple 2d')
    parser.add_argument('--world-size', nargs='+', type=float, default=np.pi, help='boundary of world')
    parser.add_argument('--reproducible', default=False, action='store_true', help='use seed bundled with trained model')

    args = parser.parse_args()
    print(args)
    main(args)
