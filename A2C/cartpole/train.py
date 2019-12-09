import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import gym
from constants import *
from ops import *
from agent import A2Cagent
import matplotlib.pyplot as plt
from rendering import Graph

env = gym.make(ENV_NAME)
agent = A2Cagent()
graph = Graph(4,200)

fig,axs = plt.subplots(1)
plt.ion()

for e in range(EPISODES):
	s = env.reset()
	s = prep_state(s)
	total_r = 0
	for t in range(TIME_LIMIT):
		env.render()
		# Choose an action based on state
		a, log_p, ent, v = agent.act(s)
		a = a.squeeze().item()
		# Take action
		s_new,r,done,info = env.step(a)
		total_r += r
		if done: break
		s_new = prep_state(s_new)
		# Get advantages
		adv = agent.get_adv(s,s_new,r)
		# Add to agents memory
		agent.remember([s, s_new, r, adv, log_p, a, done])
		s = s_new
		# Train agent
		p_loss, v_loss, ent =  agent.replay(BATCH_SIZE)
	print(total_r)
	graph.add_data([e,p_loss,v_loss,total_r])
	plt.pause(0.0001)
	graph.draw_graph(axs)
	plt.draw()
