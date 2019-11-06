from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		
		self.layer1 = nn.Linear(STATE_SIZE,128)
		self.layer2 = nn.Linear(128,128)
		self.layer3 = nn.Linear(128,NUM_ACTIONS)

	def forward(self, x):
		x = self.layer1(x)
		x = F.relu(x)
		x = self.layer2(x)
		x = F.relu(x)
		x = self.layer3(x)

		return x
