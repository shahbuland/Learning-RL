import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import Actor, Critic
from constants import *

class A3Cagent(nn.Module):
	def __init__(self):
		super(A3Cagent,self).__init__()
		
		self.A, self.C = Actor(), Critic()
		
		if USE_CUDA:
			self.A.cuda()
			self.C.cuda()

		self.opt = torch.optim.Adam(self.parameters(),lr = LEARNING_RATE)

	def forward(self, x):
		value = self.C(x)
		pi = self.A(x)

		return value, pi
