import torch
import numpy as np
import gym
from agent import A3Cagent
from rendering import Graph
from ops import *
from memory import Memory
import matplotlib.pyplot as plt


# Rendering stuff (for plotting data)
plt.ion()
fig,axs = plt.subplots(1)
graph = Graph(3, 400)

env = gym.make(ENV_NAME)
agent = A3Cagent()

entropy_total = 0

for e in range(EPISODES):
	s = env.reset()
	s = prep_state(s)
	total_r = 0

	for t in range(TIME_LIMIT):
		v, pi = agent(s) # Get value and policy
		v = v.squeeze().item() # Get actual value
		dist = pi.cpu().detach().numpy() # Get action distribution
		#env.render()
		# Sample action from action distribution
		a = np.random.choice(ACTION_SIZE, p = np.squeeze(dist)) 
	
		log_pi = torch.log(pi.squeeze(0))[a] # Log prob of the action we chose
		entropy = -np.sum(np.mean(dist) * np.log(dist))
		agent.total_entropy += entropy

		# Env step
		new_s, r, done, info = env.step(a)
		new_s = prep_state(new_s)
		total_r += r

		# Update memory
		agent.remember(log_pi,v,r)

		s = new_s
		
		if done: break

	# When episode ends, we update model
	ac_loss = agent.replay(new_s) # Replay needs terminal state as input

	# Graph stuff
	graph.add_data([e,ac_loss/100,total_r])
	plt.pause(0.0001)
	graph.draw_graph(axs)
	plt.draw()
