from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		#Input is 210 x 160 x 3	
		self.conv1 = nn.Conv2d(1,64,4,(5,4),1) # 64 x 42 x 40
		self.conv2 = nn.Conv2d(64,128,4,3,1) # 128 x 14 x 13
		self.conv3 = nn.Conv2d(128,256,4,2,1) # 256 x 7 x 6
		self.fc1 = nn.Linear(256*7*6,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128, NUM_ACTIONS)

	def forward(self, x):
		if len(list(x.shape)) == 3:
			x = x.expand(1,-1,-1,-1)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = x.view(-1, 256*7*6)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x
