import matplotlib.pyplot as plt
import torch
import numpy as np

# This script is a collection of methods useful for visualizing
# With matplotlib
# I separated it from the main training loop as it brings too much clutter
# into it otherwise

# Used with reconstructed state
# Shows us how good the agents embedding is so far
def render_state(axs,s_rec):
	axs.cla()
	s_rec = s_rec.cpu().detach().numpy()
	s_rec = np.squeeze(s_rec)
	axs.imshow(s_rec,cmap='gray')

colours = ['red','green','blue','orange'] # List of colours for plots

# Graph object for tracking data
# We always assume first number in data is x axis, everything after is y
class Graph:
	# param_number is how many things graph should track
	# max_len is too keep graph from getting too long
	def __init__(self, param_number, max_len):
		self.param_number = param_number
		self.values = [[] for _ in range(param_number)]	
		self.max_len = max_len

	# Adds data to graph
	# Assumes data is param_number length list
	def add_data(self,data):
		for i in range(self.param_number):
			self.values[i].append(data[i])
		if(len(self.values[0]) > 100):
			for i in range(self.param_number):
				del self.values[i][0]

	def draw_graph(self,axs):
		data = [np.asarray(self.values[i]) for i in range(self.param_number)]
		axs.cla()
		horizontal = data[0]
		for i in range(1,self.param_number):
			axs.plot(horizontal, data[i], color=colours[i-1])

	
	
