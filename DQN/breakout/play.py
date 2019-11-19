import numpy as np
import gym
import torch
from agent import Agent
import time
from constants import *
from ops import prep_state

time_limit = 9999
FPS = 30

SPF = 1/FPS

env = gym.make(ENV_NAME)
agent = Agent()

try:
	agent.model.load_state_dict(torch.load("params.pt"))
	print("Loaded checkpoint")
except:
	print("Could not load checkpoint")

while True:
	total_r = 0
	s = env.reset()
	s = prep_state(s)

	for i in range(TIME_LIMIT):
		
	#	time.sleep(SPF)
		env.render()
	
		a = agent.act(s, explore=False)

		s,r,done,_ = env.step(a)
		s = prep_state(s)
		total_r += r

		if done: break

	print(total_r)		
