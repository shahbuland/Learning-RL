import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from ops import prep_state
from rendering import render_state
from constants import *

env = gym.make(ENV_NAME)
fig,axs = plt.subplots()
s = env.reset()

while True:
	fig,axs = plt.subplots()
	s,_,_,_ = env.step(1)
	s = prep_state(s)
	render_state(axs,s)
	plt.show()
	plt.close()
	plt.cla()
