from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		#Input is STATE_SIZE	
		self.fc1 = nn.Linear(STATE_SIZE,256)
		self.fc2 = nn.Linear(256, 16)
		self.fc3 = nn.Linear(16, NUM_ACTIONS)
	def forward(self, x):

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)

		return x

