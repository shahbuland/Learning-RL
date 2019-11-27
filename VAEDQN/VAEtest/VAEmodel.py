from torch import nn
from torch.nn import functional as F
from models import Encoder, Decoder
from constants import *
import random
from losses import VAE_LOSS
import torch

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		
		self.E, self.D = Encoder(), Decoder()
		self.memory = []
		self.memory_num = 0
	
		self.opt = torch.optim.Adam(self.parameters(), lr = VAE_LEARNING_RATE) 

	def remember(self, s):
		if self.memory_num >= EXP_BUFFER_MAX:
			del self.memory[0]
		else:
			self.memory_num += 1 
		self.memory.append(s)
		
	def encode(self, x):
		return self.E(x)[2]
	
	def decode(self,x):
		return self.D(x)

	def forward(self, x):
		mu, logvar, z = self.E(x)
		rec_x = self.D(z)

		return mu, logvar, rec_x

	# Trains on single batch 
	def replay(self,batch_size = VAE_BATCH_SIZE):

		# Don't train if buffer isn't filled
		if len(self.memory) < VAE_BATCH_SIZE:
			return 0

		# Sample batch randomly
		batch = random.sample(self.memory, batch_size)
		s_batch = torch.cat(batch)
		output_batch = self.forward(s_batch)
		self.opt.zero_grad()
		loss = VAE_LOSS(output_batch, s_batch)
		loss.backward()
		self.opt.step()
		return loss.item()	
