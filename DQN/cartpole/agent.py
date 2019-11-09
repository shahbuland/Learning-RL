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
		self.model = DQN()
		self.exp_buffer = [] # exp buffer
		self.exp_number = 0 # size of exp buffer so far

		self.opt = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
		self.loss = nn.MSELoss()
	# Make an action given a state
	def act(self, state, explore = True):
		if explore and np.random.rand() <= EPSILON:
			# Act randomly
			a = np.random.randint(2)
			return a
		
		# Send state to model
		state = torch.from_numpy(state).float()
		a_vec = self.model(state)
		a = int(torch.argmax(a_vec))
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

		# Convert numpy arrays to tensor
		exp[0] = torch.from_numpy(exp[0]).float()
		if exp[4] == False: exp[3] = torch.from_numpy(exp[3]).float()

		self.exp_buffer.append(exp)

	# Replay gets batch and trains on it
	def replay(self, batch_size):
		# If experience buffer isn't right size yet, don't do anything
		if self.exp_number < MIN_BUFFER_SIZE: return
		# Get batch from experience_buffer
		batch_ind = list(torch.randint(self.exp_number, (batch_size,)).numpy())
		batch = get_sublist(self.exp_buffer, batch_ind)
		q_loss = 0	
		# Go through samples
		for s, a, r, s_new, done in batch:
			
			if done:
				Q_val = r
			else:
				Q_val = r + GAMMA * torch.max(self.model(s_new))
	
			self.opt.zero_grad()
	
			Q_pred = self.model(s)
			Q_targ = self.model(s)
			Q_targ[a] = Q_val
	
			myloss = self.loss(Q_pred, Q_targ)
			myloss.backward()
			q_loss += myloss.item()
			self.opt.step()

		global EPSILON
		if EPSILON > EPSILON_MIN:
			EPSILON *= EPSILON_DECAY 

		return q_loss
