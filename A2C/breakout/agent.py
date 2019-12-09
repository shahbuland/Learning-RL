import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import Actor, Critic, FeatureExtractor
from constants import *

class A3Cagent(nn.Module):
	def __init__(self):
		super(A3Cagent,self).__init__()
	
		self.FE = FeatureExtractor()
		self.A, self.C = Actor(), Critic()
		
		if USE_CUDA:
			self.A.cuda()
			self.C.cuda()
			self.FE.cuda()

		self.opt = torch.optim.Adam(self.parameters(),lr = LEARNING_RATE)

		self.mem = [[],[],[]] # Stores log_probs, values, rewards during episode
		self.total_entropy = 0

	# Adds to memory
	def remember(self, log_p, v, r):
		for i,data in enumerate([log_p,v,r]):
			self.mem[i].append(data)

	def forward(self, x):
		x = self.FE(x)
		x = x.view(-1, 512*5*5)
		
		value = self.C(x)
		pi = self.A(x)

		return value, pi

	# Requires terminal state val to train
	def replay(self, s_last):
		# Load memory
		log_P, V, R = self.mem
		# We work backwards through episode
		qval,_ = self.forward(s_last)
		qval = qval.squeeze().item()
		# Create tensor for all qvals
		Q = torch.zeros(len(log_P))
		
		# Go backwards to fill Q
		for t in reversed(range(len(log_P))):
			qval = self.mem[2][t] + GAMMA * qval
			Q[t] = qval

		# Turn all memory into tensors
		log_P = torch.stack(log_P)
		Q = torch.FloatTensor(Q)
		V = torch.FloatTensor(V)

		# To GPU
		if USE_CUDA:
			log_P = log_P.cuda()
			Q = Q.cuda()
			V = V.cuda()

		Adv = Q - V
		A_Loss = (-log_P * Adv).mean()
		C_Loss = 0.5*Adv.pow(2).mean()
		ac_loss = A_Loss + C_Loss + ENT_WEIGHT*self.total_entropy
		
		self.opt.zero_grad()
		ac_loss.backward()
		self.opt.step()
 			
		# Clear memory after training step
		self.mem = [[],[],[]]
		return ac_loss
