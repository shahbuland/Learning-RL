from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		#Input is 1 x 84 x 84	
		self.conv1 = nn.Conv2d(1,32,4,4,1) # 64 x 21 x 21
		self.conv2 = nn.Conv2d(32,64,4,2,1) # 128 x 10 x 10
		self.conv3 = nn.Conv2d(64,128,4,2,1) # 256 x 5 x 5
		self.fc1 = nn.Linear(128*5*5,512)
		self.fc2 = nn.Linear(512, NUM_ACTIONS)

	def forward(self, x):
		if len(list(x.shape)) == 3:
			x = x.expand(1,-1,-1,-1)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = x.view(-1, 128*5*5)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x
