from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(STATE_SIZE,128)
		self.fc2 = nn.Linear(128, NUM_ACTIONS)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x
