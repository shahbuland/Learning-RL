import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from constants import *

# For testing purposes
def render_state(s):
	plt.imshow(s,cmap='gray')
	plt.show()
	plt.close()

def prep_state(s):
	s = np.moveaxis(s,2,0)
	s_gray = s[0]+s[1]+s[2]
	# Down size to 86 x 86 
	# First convert to PIL image
	t = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((84,84)),
		transforms.ToTensor()])
	s_gray = t(s_gray).float() # This adds axis too	
	
	return s_gray.cuda() if USE_CUDA else s_gray
