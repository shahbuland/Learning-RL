import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import ops
from torch.distributions import Normal, Categorical

from constants import *
import layers as L
import ops

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()		

		self.conv_layers = nn.ModuleList()

		# Conv layers

		# 3 x 84 x 84 ->
		self.conv_layers.append(L.ConvBlock(CHANNELS, 32, 8, 4))
		self.conv_layers.append(L.ConvBlock(32, 64, 4, 2))
		self.conv_layers.append(L.ConvBlock(64, 64, 3, 1))
		# -> 64 x 7 x 7

		# Flatten and add FC Layers
		
		self.fc_layers = nn.ModuleList()
		
		self.fc_layers.append(nn.Linear(64 * 7 * 7, 256))
		self.fc_layers.append(nn.ReLU())
		self.fc_layers.append(nn.Linear(256, 448))
		self.fc_layers.append(nn.ReLU())

	def forward(self, x):
		if(torch.min(x) < 0 or torch.max(x) > 1): 
			print("Warning: input not normalized")
		# Forward through conv layers
		for conv in self.conv_layers:
			x = conv(x)
		# Flatten		
		x = x.view(-1, 64 * 7 * 7)

		# Two dense layers
		for fc in self.fc_layers:
			x = fc(x)
		
		return x

class ActorCritic(nn.Module):
	def __init__(self, use_conv = True, input_size = None):
		super(ActorCritic, self).__init__()
		self.use_conv = use_conv

		# Initialize conv layers, then actor and critic layers
		if self.use_conv:
			self.conv = ConvNet()

		# Size of input to FC layers
		input_size = 448 if input_size is None else input_size
		
		# Actor
		self.actor = nn.Sequential(
			nn.Linear(448 if input_size is None else input_size, 448),
			nn.ReLU(),
			nn.Linear(448, OUTPUT_SIZE)
		)

		# Critic
		self.critic = nn.Sequential(
			nn.Linear(448 if input_size is None else input_size, 448),
			nn.ReLU(),
			nn.Linear(448, 1)
		)
	
		# Weight initialization
		self.apply(self.init_weights)
	
	# Forward pass, returns action probabilities and value of state
	def forward(self, x):
		# Get features
		if self.use_conv: x = self.conv(x)

		# Actions
		act = self.actor(x)
		act_probs = F.softmax(act, dim = 0)
		
		# Value
		val = self.critic(x)
		# Return distribution as its easier for training
		return Categorical(act_probs), val 

	# Weight init
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean = 0, std = 0.1)
			nn.init.constant_(m.bias, 0.1)

		elif isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight)

	# Basic functionality for loading and saving weights
	def load_weights(self, path):
		try:
			self.load_state_dict(torch.load(path))
			print("Weights loaded successfully")
		except:
			print("Could not load weights")

	def save_weights(self, path):
		torch.save(self.state_dict(), path)
		print("Weights saved")
