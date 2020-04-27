import torch
from torch import nn
import torch.nn.functional as F

from constants import *

# Conv block with conv2d, activation and optons for batch norma and pooling
class ConvBlock(nn.Module):
	def __init__(self, fi, fo, k=4, s=1, p=0, act = None, bn = False, pool = False):
		super(ConvBlock, self).__init__()
			
		self.layers = nn.ModuleList()
		
		self.layers.append(nn.Conv2d(fi, fo, k, s, p))
		if pool:
			self.layers.append(nn.MaxPool2d(2))
		if act is not None:
			self.layers.append(act)
		if bn:
			self.layers.append(nn.BatchNorm2d(fo))
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

# Layer that converts conv output into a vector for input 
# To fully connected layers
# Created to provide simpler and general alternative to .view
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	# Option for sequence input
	def forward(self, x, seq = False):
		# Get listed shape of input
		shape = list(x.shape)
		def prod(L):
			if not L: return 0
			res = 1
			for i in L:
				res *= i
			return res

		if not seq:
			# Shape for input to FC
			dim = prod(shape[1:])
			x = x.view(-1, dim)

		if seq:
			dim = prod(shape[2:])
			x = x.view(-1,-1,dim)
			
		return x
