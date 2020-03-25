from torch import nn
import torch
from constants import *
import torch.nn.functional as F

# Actor network (Gets P(a_t | s_t) )
class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
	
		self.fc1 = nn.Linear(STATE_SIZE,256)
		self.fc2 = nn.Linear(256,ACTION_SIZE)

	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.softmax(x)
		return x

# Critic network (Gets V(a_t | s_t) )
class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		
		self.fc1 = nn.Linear(STATE_SIZE,256)
		self.fc2 = nn.Linear(256,1)

	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x
