import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *
from models import Actor, Critic

class A2Cagent(nn.Module):
	def __init__(self):
		super(A2Cagent.self).__init__()

		self.A, self.C = Actor(), Critic()
		self.opt = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

	def forward(self, x):
		a_p = self.A(x)
		v = self.C(x)
		return a_p, v
				
	def act(self, s, action=None):
		prob, v = self.forward(s)
		dist = Categorical(prob)
		if action is None:
			action = dist.sample()
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		return action,log_prob, entropy, v.squeeze()

	# Replay S states, A actions, R Rewards, Adv advantages
	def replay(self, log_p_old, S, A, R, Adv):
		a, log_p, ent, v = self.act(S, A)
		
		# A2C loss function for actor
		p_loss_ratio = torch.exp(log_p - log_p_old)
		p_loss_1 = p_loss_ratio * Adv
		p_loss_2 = torch.clamp(p_loss_ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * Adv
		p_loss = -torch.min(p_loss_1,  p_loss_2).mean()
		
		# MSE for Critic Loss
		v_loss = 0.5*(R - v).pow(2).mean()
		ent = ent.mean()

		self.opt.zero_grad()
		(p_loss + v_loss - BETA*ent).backward()
		nn.utils.clip_grad_norm_(self.parameters(), 5)
		self.opt.step()

		return p_loss, v_loss, ent	
