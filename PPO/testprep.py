import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from ops import prep_obs

# Tests the prep obs function

env = gym.make("BreakoutNoFrameskip-v4")
s = env.reset()
t = prep_obs(s)
print(t.shape)
a = t.detach().cpu().numpy()
a = np.squeeze(a)
plt.imshow(a,cmap="gray")
plt.show()
plt.close()
