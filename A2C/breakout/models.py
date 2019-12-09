from torch import nn
import torch
from constants import *
import torch.nn.functional as F

# Gets feature map for actor and critic
class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()

		# Input shape is 1 x 84 x 84
		self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
		self.conv2 = nn.Conv2d(64,128,4,2,1)
		self.conv3 = nn.Conv2d(128,256,4,2,1)
		self.conv4 = nn.Conv2d(256,512,4,2,1)
		# Output shape is 512 x 5 x 5

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = F.relu(x)
		
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
		x = F.softmax(x)
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
