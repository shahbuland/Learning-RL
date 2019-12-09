import numpy as np
import random
import torch
from constants import *

# Storage structure: [log_p, values, rewards, actions]
class Memory:
	# Default value of -1 means no exp buffer cap
	def __init__(self, max_size=-1):
		self.size = 0
		self.max = max_size
		self.mem = []
		

	def clear(self):
		self.size = 0
		self.mem = []

	def add(self, exp):
		# If max is -1 we just say there is no max
		if not self.max == -1 and self.size == self.max:
			del self.mem[0]
		else:
			self.size += 1
		self.mem.append(exp)

	def get_batch(self, size):
		batch = random.sample(self.mem, size)
		log_p, v, r, a =  zip(*batch)

		PRINT = False

		if PRINT:	
			for i in [log_p,v,r,a]:
				print(type(i))
		
		assert 1==2
		return log_p,v,r,a

	def get_full(self):
		log_p, v, r, a = zip(*self.mem)

		PRINT = True		
		if PRINT:
			for i in [log_p,v,r,a]:
				print(type(i))

		assert 1 ==2
		return log_p,v,r,a
