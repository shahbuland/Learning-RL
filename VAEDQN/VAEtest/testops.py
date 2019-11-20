from ops import prep_state
import numpy as np
import torch
import gym

env = gym.make('Breakout-v0')
s = env.reset()
s_prep = prep_state(s)
print(s_prep.shape)
