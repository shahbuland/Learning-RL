import numpy as np
import random
import torch
from constants import *

class Memory:
	def __init__(self, max_size):
		self.size = 0
		self.max = max_size
		self.mem = []

	def add(self, exp):
		if self.size == self.max:
			del self.mem[0]
		else:
			self.size += 1
		self.mem.append(exp)

	# Returns [s, s_new, r, adv, log_p, a, done]
	def get_batch(self, size):
		batch = random.sample(self.mem, size)
		s,s_new, r, adv, log_p, a, done =  zip(*batch)
		s = torch.cat(s)
		s_new = torch.cat(s_new)
		r = torch.tensor(r).float().unsqueeze(1)
		adv = torch.tensor(adv).float().unsqueeze(1)
		log_p = torch.cat(log_p).unsqueeze(1)
		a = torch.tensor(a).float().unsqueeze(1)

		if USE_CUDA:
			r = r.cuda()
			adv = adv.cuda()
			a = a.cuda()
	
		PRINT = False

		if PRINT:	
			for i in [s,s_new,r,adv,log_p,a]:
				print(i.shape)
	
		return s,s_new, r, adv, log_p, a, done		
