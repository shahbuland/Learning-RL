from constants import *
from models import DQN
import numpy as np
from torch import nn
import torch
from memory import Memory

class Agent:
	def __init__(self):
		self.model, self.target = DQN(), DQN()
		if USE_CUDA:
			self.model.cuda()
			self.target.cuda()

		self.exp_buffer = Memory()
		self.exp_number = 0 # size of exp buffer so far
		self.param_updates = 0 # track how many times params updated

		self.opt = torch.optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)
		self.loss = nn.SmoothL1Loss()

	# Make an action given a state
	def act(self, state, explore=True):
		if explore and np.random.rand() <= EPSILON:
			# Act randomly
			a = np.random.randint(NUM_ACTIONS)
		else:
			# Send state to model
			a_vec = self.model(state)
			a = int(torch.argmax(torch.squeeze(a_vec)))
	
		return a
		 
	# clear the buffer
	def clear_exp_buffer(self):
		self.exp_buffer = Memory()
		self.exp_number = 0
	
	# Add experience to exp buffer
	def add_exp(self, exp):
		self.exp_buffer.add(exp)
		self.exp_number += 1
	
	# Replay gets batch and trains on it
	def replay(self, batch_size):
		q_loss = 0
		# If experience buffer isn't right size yet, don't do anything
		if self.exp_number < MIN_BUFFER_SIZE: return
		# Get batch from experience_buffer
		batch = self.exp_buffer.get_batch(batch_size)
		
		s, a, r, s_new,_ = zip(*batch)
		s_new = s_new[:-1] # Remove last item (it is 'None')
		# First turn batch into something we can run through model
		s = torch.cat(s)
		a = torch.LongTensor(a).unsqueeze(1)
		r = torch.FloatTensor(r).unsqueeze(1)
		s_new = torch.cat(s_new)
		
		#print(a.shape,r.shape, s.shape, s_new.shape)
		if USE_CUDA:
			a = a.cuda()
			r = r.cuda()

		# Get q vals for s (what model outputted) from a
		# .gather gets us q value for specific action a
		pred_q_vals = self.model(s).gather(1,a)

		# Having chosen a in s,
		# What is the highest possible reward we can get from s_new?
		# We add q of performing a in s then add best q from next state
		# cat 0 to end for the terminal state
		s_new_q_vals = self.target(s_new).max(1)[0]
		zero = torch.FloatTensor(0)
		if USE_CUDA: zero = zero.cuda()

		s_new_q_vals = torch.cat((s_new_q_vals, zero))
		exp_q_vals = r + s_new_q_vals*GAMMA
		
		myloss = self.loss(pred_q_vals, exp_q_vals)
		self.opt.zero_grad()
		myloss.backward()
		self.opt.step()
		
		if WEIGHT_CLIPPING:
			for param in self.model.parameters():
				param.grad.data.clamp_(-1,1) # Weight clipping avoids exploding gradients
	
		if self.param_updates % TARGET_UPDATE_INTERVAL == 0:
			self.target.load_state_dict(self.model.state_dict())

		self.param_updates += 1

		global EPSILON
		if EPSILON > EPSILON_MIN:
			EPSILON *= EPSILON_DECAY 

		return myloss.item()
