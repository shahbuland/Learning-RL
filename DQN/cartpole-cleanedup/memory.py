from constants import *
import numpy as np
import torch
import random

class Memory:
	def __init__(self):
		self.exp_buffer = []
	
	def add(self, exp):
		self.exp_buffer.append(exp)
		if len(self.exp_buffer) > MAX_BUFFER_SIZE:
			del self.exp_buffer[0]

	def get_batch(self, size):
		return random.sample(self.exp_buffer, size)
