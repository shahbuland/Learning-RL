import gym
from agent import Agent
from constants import *
import torch
import numpy as np
import time
from ops import prep_state

env = gym.make(ENV_NAME)
agent = Agent()

# Try loading previous agent
if LOAD_CHECKPOINTS:
	try:
		agent.model.load_state_dict(torch.load("params.pt"))
		print("Loaded checkpoint")
	except:
		print("Could not load checkpoint")

scores = []

step = 0

for e in range(EPISODES):
	total_r = 0
	s = env.reset()
	s = prep_state(s)
	
	for t in range(TIME_LIMIT):

		env.render()

		a = agent.act(s)

		s_new, r, done, _ = env.step(a)
		total_r += r
		
		s_new = prep_state(s_new)
		agent.add_exp([s, a, r, s_new, done])
		
		s = s_new
	
		# Train	
		if (step+1) % TRAINING_INTERVAL == 0:
			agent.replay(32)
		step += 1
	
		if done: break
	scores.append(total_r)

	# Save weights
	if (e+1) % CHECKPOINT_INTERVAL == 0:
		torch.save(agent.model.state_dict(), "params.pt")

	print("Episode", e, "| Reward:", total_r, "Avg Reward:", np.mean(np.asarray(scores)))
