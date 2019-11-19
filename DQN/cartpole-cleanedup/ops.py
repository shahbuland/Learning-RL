from constants import *
import numpy as np
import torch

def prep_state(s):
	s = torch.from_numpy(s).float().unsqueeze(0)
	if USE_CUDA: s = s.cuda()
	return s 
