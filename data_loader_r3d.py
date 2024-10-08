import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
#import nltk
#from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import gc

#N=number of environments; NP=Number of Paths
def load_train_dataset(N=100,NP=4000,folder='../milestone2/',s=2):
	# load data as [path]
	# for each path, it is
	# [[input],[target],[env_id]]
	print('load 3d data...')
	obs = []
	# add start s
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile(folder+'obc'+str(i+s)+'.dat')
		obs.append(temp)
	obs = np.array(obs)

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'path/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.loadtxt(fname, unpack=True)
				path=path.transpose()
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'path/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.loadtxt(fname, unpack=True)
				path=path.transpose()
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]



	dataset=[]
	targets=[]
	env_indices=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
				for m in range(0, path_lengths[i][j]-1):
					data = np.concatenate((paths[i][j][m], paths[i][j][path_lengths[i][j]-1]) ).astype(np.float32)
					targets.append(paths[i][j][m+1])
					dataset.append(data)
					env_indices.append(i)
	# only return raw data (in order), follow below to randomly shuffle
	data=list(zip(dataset,targets,env_indices))
	random.shuffle(data)
	dataset,targets,env_indices=list(zip(*data))
	dataset = list(dataset)
	targets = list(targets)
	env_indices = list(env_indices)
	return obs, dataset, targets, env_indices
	# data=list(zip(dataset,targets))
	# random.shuffle(data)
	# dataset,targets=list(zip(*data))
	# dataset and targets are both list
	# here the first item of data is index in obs
	# return obs, list(zip(*data))

def load_obs_list(env_id, folder='../data/simple'):
	obc = np.zeros((10,3),dtype=np.float32)
	temp = np.fromfile(folder + 'obs.dat')
	obs = temp.reshape(len(temp)//3,3)

	temp = np.fromfile(folder+'obs_perm2.dat',np.int32)
	perm=temp.reshape(184756,10)

	for j in range(0,10):
		for k in range(0,3):
			obc[j][k] = obs[perm[env_id][j]][k]
	# returns a list of 3d obstacles, represented by their centers (x,y,z)
	return obc

#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000, folder='../data/simple/'):
	obc=np.zeros((N,10,3),dtype=np.float32)
	temp=np.fromfile(folder+'obs.dat')
	obs=temp.reshape(len(temp)//3,3)

	temp=np.fromfile(folder+'obs_perm2.dat',np.int32)
	perm=temp.reshape(184756,10)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,10):
			for k in range(0,3):
				obc[i][j][k]=obs[perm[i+s][j]][k]


	obs = []
	k=0
	for i in range(s,s+N):
		temp=np.fromfile(folder+'obc'+str(i)+'.dat')
		obs.append(temp)
	obs = np.array(obs).astype(np.float32)
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'path/path'+str(sp+j)+'.dat'
			if os.path.isfile(fname):
				path=np.loadtxt(fname, unpack=True)
				path=path.transpose()
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'path/path'+str(sp+j)+'.dat'
			if os.path.isfile(fname):
				path=np.loadtxt(fname, unpack=True)
				path=path.transpose()
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]

	return 	obc,obs,paths,path_lengths
