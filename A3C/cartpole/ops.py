from constants import *
import numpy as np
import torch

def prep_state(s):
	s = torch.from_numpy(s).unsqueeze(0).float()
	if USE_CUDA: s = s.cuda()
	return s
