from constants import *
from models import DQN
import numpy as np
from torch import nn
import torch

# Get sublist of A elements i_0 through i_n from L = [i_0 ... i_n]
def get_sublist(A, L):
	subL = []
	for i in L:
		subL.append(A[i])
	return subL

# Exp is [state, action, reward, state_next, done]

class Agent:
	def __init__(self):
		self.model, self.target = DQN(), DQN()
		if USE_CUDA:
			self.model.cuda()
			self.target.cuda()

		self.exp_buffer = [] # exp buffer
		self.exp_number = 0 # size of exp buffer so far

		self.opt = torch.optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)
		self.loss = nn.SmoothL1Loss()

	# Make an action given a state
	def act(self, state, explore=True):
		if explore and np.random.rand() <= EPSILON:
			# Act randomly
			a = np.random.randint(NUM_ACTIONS)
			return a
		
		# Send state to model
		a_vec = self.model(state)
		a = int(torch.argmax(torch.squeeze(a_vec)))
		return a
		 
	# clear the buffer
	def clear_exp_buffer(self):
		self.exp_buffer = []
		self.exp_number = 0
	
	# Add experience to exp buffer
	def add_exp(self, exp):
		if self.exp_number == MAX_BUFFER_SIZE:
			del self.exp_buffer[0]
		else:
			self.exp_number += 1

		self.exp_buffer.append(exp)

	# Replay gets batch and trains on it
	def replay(self, batch_size):
		# If experience buffer isn't right size yet, don't do anything
		if self.exp_number < MIN_BUFFER_SIZE: return
		# Get batch from experience_buffer
		batch_ind = list(torch.randint(self.exp_number, (batch_size,)).numpy())
		batch = get_sublist(self.exp_buffer, batch_ind)
		q_loss = 0

		# First turn batch into something we can run through model
		s = torch.cat([exp[0] for exp in batch]).unsqueeze(1)
		a = torch.LongTensor([exp[1] for exp in batch]).unsqueeze(1)
		r = torch.FloatTensor([exp[2] for exp in batch])
		s_new = torch.cat([exp[3] for exp in batch][:-1]).unsqueeze(1)
		
		if USE_CUDA:
			a = a.cuda()
			r = r.cuda()

		# Get q vals for s (what model outputted) from a
		# .gather gets us q value for specific action a
		pred_q_vals = self.model(s).gather(1,a).squeeze(1)

		# Having chosen a in s,
		# What is the highest possible reward we can get from s_new?
		# We add q of performing a in s then add best q from next state
		# cat 0 to end for the terminal state
		s_new_q_vals = self.target(s_new).max(1)[0]

		zero = torch.FloatTensor([0])
		if USE_CUDA: zero = zero.cuda()

		s_new_q_vals = torch.cat((s_new_q_vals, zero))
		exp_q_vals = r + s_new_q_vals*GAMMA
		
		myloss = nn.functional.smooth_l1_loss(pred_q_vals, exp_q_vals)
		self.opt.zero_grad()
		myloss.backward()
		for param in self.model.parameters():
			param.grad.data.clamp_(-1,1) # Weight clipping avoids exploding gradients
		self.opt.step()

		global EPSILON
		if EPSILON > EPSILON_MIN:
			EPSILON *= EPSILON_DECAY 

		return myloss.item()
