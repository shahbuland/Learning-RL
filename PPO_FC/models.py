import torch
from torch.distributions.categorical import Categorical
from torch import nn
import torch.nn.functional as F
from constants import *

class ActorCritic(nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()

		# First define actor and critic
		self.actor = nn.Sequential(nn.Linear(STATE_SIZE, 256), nn.Linear(256, ACTION_SIZE))
		self.critic = nn.Sequential(nn.Linear(STATE_SIZE, 256), nn.Linear(256, 1))

	# Run forward to get action probs and state value
	def forward(self, s):
		act_probs = self.actor(s)
		val = self.critic(s)
		return act_probs, val

	# Get an action (assumes s is single state
	def act(self, s):
		assert list(s.shape)[0] == 1
		act_probs, _ = self.forward(s)
		act_probs = act_probs[0] # Remove the batch axis
		dist = Categorical(act_probs)
		return dist.sample()

# Same as above but for continuous actions spaces
class ContActorCritic(nn.Module):
	def __init__(self):
		super(ContActorCritic, self).__init__()

		# This part is same
		self.actor = nn.Sequential(nn.Linear(STATE_SIZE, 256))
		self.critic = nn.Sequential(nn.Linear(STATE_SIZE, 256), nn.Linear(256, 1))
		# However now the actor should output parameters for normal distribution
		# Thus the policy action is continous rv from normal
		self.mu = nn.Linear(256, ACTION_SIZE)
		self.std = nn.Linear(256, ACTION_SIZE) # We will have to relu this

	def forward(self, x):
		return x
