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

mem = Memory()

entropy_total = 0

for e in range(EPISODES):
	s = env.reset()
	s = prep_state(s)
	total_r = 0

	log_probs = []
	values = []
	rewards = []

	for t in range(TIME_LIMIT):
		v, pi = agent(s) # Get value and policy
		v = v.squeeze().item() # Get actual value
		dist = pi.cpu().detach().numpy() # Get action distribution
		env.render()
		# Sample action from action distribution
		a = np.random.choice(ACTION_SIZE, p = np.squeeze(dist)) 
	
		log_pi = torch.log(pi.squeeze(0))[a] # Log prob of the action we chose
		entropy = -np.sum(np.mean(dist) * np.log(dist))
		entropy_total += entropy

		# Env step
		new_s, r, done, info = env.step(a)
		new_s = prep_state(new_s)
		total_r += r

		# Update memory
		rewards.append(r)
		values.append(v)
		log_probs.append(log_pi)

		s = new_s
		
		if done: break

	# When episode ends, we update model
	
	# Begin by making q val for terminal state
	# Then work backwards
	Qval, _ = agent(new_s)
	Qval = Qval.squeeze().item()

	Qvals = torch.zeros(len(values))

	for t in reversed(range(len(rewards))):
		# Reward at t
		Qval = rewards[t] + GAMMA * Qval
		Qvals[t] = Qval
	
	values = torch.FloatTensor(values)
	Qvals = torch.FloatTensor(Qvals)
	log_probs = torch.stack(log_probs)
	
	if USE_CUDA:
		values = values.cuda()
		Qvals = Qvals.cuda()
		log_probs = log_probs.cuda()

	# Update model
	advantage = Qvals - values
	actor_loss = (-log_probs * advantage).mean()
	critic_loss = 0.5 * advantage.pow(2).mean()
	ac_loss = actor_loss + critic_loss + 0.001*entropy_total

	agent.opt.zero_grad()
	ac_loss.backward()
	agent.opt.step()

	# Graph stuff
	graph.add_data([e,ac_loss/100,total_r])
	plt.pause(0.0001)
	graph.draw_graph(axs)
	plt.draw()
