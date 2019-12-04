import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import gym
from constants import *
from ops import *

env = gym.make(ENV_NAME)
agent = A2Cagent()

for e in range(EPISODES):
	s = env.reset()
	s = prep_state(s)
	for t in range(TIME_LIMT):
		# Choose an action based on state
		a, log_p, ent, v = agent.act(s)
		# Take action

