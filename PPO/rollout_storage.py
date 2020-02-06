import torch

import ops

# Number of items to store 
ROLLOUT_ITEMS = 6

# Object to store rollouts 
# When returning or storing value order is always:
# Log_probs, Values, States, Actions, Rewards, Masks
class RolloutStore:
	def __init__(self):
		self.store = [[] for _ in range(ROLLOUT_ITEMS)]
 
	# Assumes what's being added is tensor so it can run torch.cat later
	def add(self,L):
		for i in range(ROLLOUT_ITEMS):
			self.store[i].append(L[i])

	# Return list of stored tensors
	def unwind(self):
		retval = [torch.cat(self.store[i]) for i in range(ROLLOUT_ITEMS)]
		return retval

	# Sets rewards to GAE returns using masks
	# [INFERS INDEX FOR REWARDS FROM LIST ABOVE]
	def update_gae(self, next_v):
		self.store[4] = ops.get_GAE(next_v, self.store[4],
									 self.store[5], self.store[1]) 
	
	# Empty and reset rollout storage
	def reset(self):
		self.store = [[] for _ in range(ROLLOUT_ITEMS)]

	
