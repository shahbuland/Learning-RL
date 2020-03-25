from torch import nn
import torch
from constants import *
import torch.nn.functional as F

# Gets feature map for actor and critic
class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()

		# Input shape is 1 x 84 x 84
		self.conv1 = nn.Conv2d(1, 32, 4, 4, 1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32,64,3,2,1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64,64,3,2,1)
		self.bn3 = nn.BatchNorm2d(64)
		# Output shape is 64 x 6 x 6

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.bn2(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.bn3(x)
	
		return x
		
# Actor network (Gets P(a_t | s_t) )
class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()

		self.fc1 = nn.Linear(STATE_SIZE,256)
		self.fc2 = nn.Linear(256,256)
		self.fc3 = nn.Linear(256,ACTION_SIZE)

	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.softmax(x,dim=1)
		return x

# Critic network (Gets V(a_t | s_t) )
class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		self.fc1 = nn.Linear(STATE_SIZE,256)
		self.fc2 = nn.Linear(256,256)
		self.fc3 = nn.Linear(256,1)

	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x
