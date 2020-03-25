from constants import *
from ops import *
import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make(ENV_NAME)

s = env.reset()
s = prep_state(s)
render_state(s)
