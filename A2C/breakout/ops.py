from constants import *
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# For testing purposes
def render_state(s):
	s = s.detach().cpu().squeeze().numpy()
	plt.imshow(s, cmap='gray')
	plt.show()
	plt.close()

def prep_state(s):
	# Move channels to front
	s = np.moveaxis(s,2,0)
	# Turn into torch tensor
	s = torch.from_numpy(s).float()
	# Normalize:
	s = s/255
	# Grayscale:
	s = (s[0] + s[1] + s[2])/3
	# Adds channel and minibatch
	s = s.unsqueeze(0).unsqueeze(0)
	# Downsize to 84x84
	s = F.interpolate(s, size = (84,84))
	
	return s.cuda() if USE_CUDA else s

# Xavier weight init
def init_weights(m):
	if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
		torch.nn.init.kaiming_uniform(m.weight)
		m.bias.data.fill_(0.01)
