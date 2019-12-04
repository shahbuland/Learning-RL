import numpy as np
import torch

def prep_state(s):
	return torch.from_numpy(s).unsqueeze(0)
