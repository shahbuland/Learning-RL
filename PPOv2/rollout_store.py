import torch

# Object for storing and unloading trajectories
# Assume trajectories are ordered:
# state, next_state, action, log prob of action, reward, done

# NOTE: Assumes trajectory inputs are fully prepared (i.e. normalized)
num_items = 6 # Number of items we need to store
class Rollout_Storage:
	def __init__(self):
		self.store, self.size = None, None
		self.flush()

	def add(trajectory):
		for i in range(num_items):
			self.store[i].append(trajectory[i])

	# samples a batch of some size from storage
	def sample(batch_size):
		# Get some indices
		ind = torch.random.randint(self.size, (batch_size,))
		# Return those indices for every item
		return [self.store[i][ind] for i in range(num_items)]

	# Empties (resets) storage
	def flush():
		self.store = [[] for _ in range(num_items)]
		self.size = 0
