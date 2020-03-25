import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from constants import *

# Attempts to convert data to model ready input
# Targ type can either be "Float" or "Long"
def prep_data(data, targType = "Float"):
	
	# Convert to tensor if array
	if type(data) == np.ndarray:
		data = torch.from_numpy(data)

	# Convert to float or long tensor
	if targType == "Float":
		data = data.float()
	elif targType == "Long":
		data = data.long()
	
	return data.cuda() if USE_CUDA else data

# Prep observation as model input
# Assumes data is direct state output from env
def prep_obs(data):

	# In case input isn't meant to be image
	if not INPUT_IMAGE:
		data = torch.Tensor(data).float()
		if USE_CUDA: 
			return data.cuda()
		else:
			return data

	# Convert to tensor if array
	if type(data) == np.ndarray:
		data = np.moveaxis(data, 2,0)
		data = torch.from_numpy(data)
	
	data = data.float()
	
	# Get a mini-batch dimension if we need it
	if len(list(data.shape)) == 3: data = data.unsqueeze(0)

	# resize
	data = F.interpolate(data, size=(IMAGE_SIZE, IMAGE_SIZE))

	data = data * (1/255)
	
	# Grayscale
	if CHANNELS == 1:
		data = data[:,0,:,:] + data[:,1,:,:] + data[:,2,:,:]
		data = data/3
		data = data.unsqueeze(1)

	return data.cuda() if USE_CUDA else data

# Gets tensor from list of tensors
# Makes the assumption that items in list are themselves tensors
# Preserves dimension, i.e.
# If x \in L has shape [1,1] and L has length [100], then result
# should have shape [100,1,1]
def list_to_Tensor(L):
	l = len(L)
	shape = list(L[0].shape)
	Y = torch.zeros([l] + shape)
	for i in range(l):
		Y[i] = L[i]
	return Y

# Quickly samples from categorical distribution created
# by probability vector
def sample_from(prob):
	dist = Categorical(prob)
	return dist.sample().item()

# make mask for GAE
def makeMask(done):
	mask = torch.FloatTensor([1 - done]).unsqueeze(1)
	if USE_CUDA: mask = mask.cuda()
	return mask

# Get general advantage estimate
# Needs next value, rewards, masks, values
def get_GAE(next_v, r, m, v):
	v = v + [next_v]
	gae = 0
	returns = []
	# Iterate backwards through rollout
	for t in reversed(range(len(r))):
		delta = r[t] + GAMMA * v[t + 1] * m[t] - v[t]
		gae = delta + GAMMA * TAU * m[t] * gae
		# Since we are iterating backwards, we have to reverse again
		# to get the returns in right order
		returns.insert(0, gae + v[t])
	return returns

# check if x contains any nan values
def isnan(x):
	return torch.max(x != x)
