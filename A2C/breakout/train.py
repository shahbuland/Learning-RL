import torch
import numpy as np
import gym
from agent import A2Cagent
from rendering import Graph, render_state
from ops import prep_state
import matplotlib.pyplot as plt
from constants import *

# Rendering stuff (for plotting data)
plt.ion()
fig,axs = plt.subplots(2)
graph = Graph(3, 400)

env = gym.make(ENV_NAME)
agent = A2Cagent()
if USE_CUDA: agent.cuda()

for e in range(EPISODES):
	s = env.reset()
	s = prep_state(s)
	total_r = 0
	for t in range(TIME_LIMIT):
		env.render()

		v, pi = agent(s) # Get value and policy
		v = v.squeeze().item() # Get actual value
		# Sample action from action distribution
		m = torch.distributions.Categorical(torch.exp(pi))
		a = m.sample()
	
		log_pi = pi.squeeze(0)[a] # Log prob of the action we chose
		entropy = -torch.sum(torch.exp(pi.detach()) * pi.detach())
		agent.total_entropy = entropy

		# Env step
		new_s, r, done, info = env.step(a)
		new_s = prep_state(new_s)
		total_r += r

		# Update memory
		agent.remember(log_pi,v,r)

		s = new_s
	
		# Step agent (trains if training interval satisfied)
		ac_loss = agent.step(new_s)

		if ac_loss is None: ac_loss = 0
		
		if done: break
		
	# Graph stuff
	graph.add_data([e*TIME_LIMIT, ac_loss, total_r])
	plt.pause(0.0001)
	graph.draw_graph(axs[0])
	plt.draw()
