import torch
from torch import nn
import layers as L
import torch.nn.functional as F
import numpy as np
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
		# Forward through conv layers
		for conv in self.conv_layers:
			x = conv(x)

		# Flatten		
		x = x.view(-1, 64, 7, 7)

		# Two dense layers
		for fc in self.fc_layers:
			x = fc(x)
		
		return x

class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()

		# Initialize conv layers, then actor and critic layers
		self.conv = ConvNet()

		# Actor
		self.actor = nn.Sequential(
			nn.Linear(448, 448),
			nn.ReLU(),
			nn.Linear(448, OUTPUT_SIZE)
		)

		# Critic
		self.critic = nn.Sequential(
			nn.Linear(448, 448),
			nn.ReLU(),
			nn.Linear(448, 1)
		)
		
		# Weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				init.orthogonal_(m.weight, np.sqrt(2))
				m.bias.data.zero_()
		
		init.orthogonal_(self.critic.weight, 0.01)
		self.critic.bias.data.zero_()

		init.orthogonal_(self.actor.weight, 0.01)
		self.actor.bias.data.zero_()
	
	# Forward pass, returns action probabilities and value of state
	def forward(self, x):
		# Get features
		x = self.conv(x)

		# Actions
		act = self.actor(x)
		act_probs = F.softmax(act, dim = 1)
		
		# Value
		val = self.critic(x)
		
		return act_probs, val

	# Takes action based on policy
	def act(self, x):
		x = self.conv(x)
		act = self.actor(x)
		act_probs = F.softmax(act, dim = 1)
		return ops.sample_from(act_probs)

	# Basic functionality for loading and saving weights
	def load_weights(self, path):
		try:
			model.load_state_dict(torch.load(path))
			print("Weights loaded successfully")
		except:
			print("Could not load weights")

	def save_weights(self, path):
		torch.save(model.state_dict(), path)
		print("Weights saved")
