import gym
from agent import Agent
from constants import *
import torch
import numpy as np

env = gym.make("CartPole-v1")
agent = Agent()
time_limit = 500

for e in range(EPISODES):
	total_r = 0
	s = env.reset()

	for t in range(time_limit):
		env.render()
		a = agent.act(s)
		s_new, r, done, _ = env.step(a)
		total_r += r
		agent.add_exp([s, a, r, s_new, done])
		s = s_new
		if done: break
	print("Episode", e, "| Reward:", total_r)
	agent.replay(32)

