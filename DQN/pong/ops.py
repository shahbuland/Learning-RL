import numpy as np
import matplotlib.pyplot as plt

# For testing purposes
def render_state(s):
	plt.imshow(s,cmap='gray')
	plt.show()
	plt.close()

def prep_state(s):
	s = np.moveaxis(s,2,0)
	s_gray = s[0]+s[1]+s[2]
	s_gray = np.expand_dims(s_gray,0)
	return s_gray
