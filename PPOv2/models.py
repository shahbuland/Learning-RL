import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from constants import *
from layers import ConvBlock, Flatten

# Encoder component
# For images, we assume input of 3x84x84
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		
		self.conv_layers == nn.ModuleList()
		self.flat = Flatten()
		
		self.conv_layers.append(ConvBlock(CHANNELS, 32, 8, 4, act = nn.ReLU()))
		self.conv_layers.append(ConvBlock(32, 64, 4, 2, act = nn.ReLU()))
		self.conv_layers.append(ConvBlock(64, 64, 3, 1, act = nn.ReLU()))

	def forward(self, x):
		for layer in self.conv_layers:
			x = layer(x)
		
		self.flat(x)
		return x
					

# Main actor critic
# Forward returns distribution (policy), and value
class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()

		if USE_CONV: self.conv = ConvNet()
		self.fc = nn.Linear(NET_INPUT, 512)
		
		# FC layers for policy and value approximator respectively
		self.pi_fc = nn.Linear(512, ACTION_DIM)
		self.val_fc = nn.Linear(512, 1)

	def forward(self, x):
		is USE_CONV: x = self.conv(x)	

		x = self.fc(x)
		x = F.relu(x)

		logits = self.pi_fc(x) # Log probabilities over actions
		val = self.val_fc(x) # Value of state
		pi = Categorical(logits.exp()) # Policy distribution

		return pi, val
