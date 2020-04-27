from constants import *
import torch
import torch.nn.functional as F

# To normalize a tensor 
def normalize(t):
	return (t - t.mean())/t.std()

# To prepare state
# State is assumed to just be a numpy array directly from env
def prep_state(s):
	s = torch.from_numpy(s).float()
	if USE_CUDA: 
		s = s.cuda()
	s = s.unsqueeze(0) # Add batch axis

	if not USE_CONV:
		# If input isn't image, this is much simpler
		return s

	# If it is an image, we need change channel to be in front (i.e. from NHWC to NCHW)
	s = s.permute(0,3,1,2)
	# Downscale to 84 x 84
	s = F.interpolate(s, size = (IMAGE_SIZE, IMAGE_SIZE))
	# From [0,255] -> [0,1]
	s *= (1/255) 
	# Grayscale
	if CHANNELS == 1:
		# Add together all colour channels then unsqueeze to add back
		# The channel dimension that this summation removes
		s = (s[:,0,:,:] + s[:,1,:,:] + s[:,2,:,:]).unsqueeze(0)

	return s

# Modified version of step to handle torch tensor I/O
# In: Gym Environment, Action (Single element long tensor)
# Out: Float Tensor, Float Tensor, Float Tensor
def mod_step(env, a):
	s, r, d, _ = env.step(a.item())

	s = prep_state(s)
	r = torch.Tensor([r]).float()
	d = (torch.ones(1) if d else torch.zeros(1)).float()
	if USE_CUDA:
		r = r.cuda()
		d = d.cuda()

	return s, r, d

# Get generalized advantage estimate from dones, rewards and values
# Also needs float for last_v to start iteration
# Assumes all input tensors are of same given size num steps
# Calculates using recursive definition of GAE with delta
# Also assumes inputs are vector tensors
def get_gae(d, r, v, last_v, num_steps):
	adv = torch.zeros(num_steps)
	last_adv = 0 # init to 0
	# Go through steps backwards to use recursive definition
	for t in reversed(range(num_steps)):
		mask = 1 - d[t]
		last_v = last_v * mask # V and Adv becomes zero if ended game
		last_adv = last_adv * mask 

		delta = r[t] + GAMMA * last_v - v[t]
		last_adv = delta + GAMMA * TAU * last_adv 
		
		adv[t] = last_adv

	return adv
