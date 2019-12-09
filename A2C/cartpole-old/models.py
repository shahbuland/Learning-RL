from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from constants import *

class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
	
		self.fc1 = nn.Linear(STATE_SIZE, ACTION_SIZE)

	def forward(self,x):		
		return F.softmax(self.fc1(x))

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		self.fc1 = nn.Linear(STATE_SIZE, 1)

	def forward(self, x):
		return self.fc1(x)
