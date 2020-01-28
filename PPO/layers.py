import torch
from torch import nn

# Filters in, filters out, kernal size, stride, padding, activation function, use batch norm?, use pooling?
class ConvBlock(nn.Module):
	def __init__(self, fi, fo, k=4, s=1, p=0, act = nn.ReLU(), bn = False, pool = False):
		super(ConvBlock, self).__init__()

		# Use list of layers to store block
		self.layers = nn.ModuleList()

		self.layers.append(nn.Conv2d(fi,fo,k,s,p))

		if pool:
			self.layers.append(nn.MaxPool2d(2))

		if act is not None:
			self.layers.append(act)
		
		if bn:
			self.layers.append(nn.BatchNorm2d(fo))

	# Forward pass just iterates through list of layers
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x
		
