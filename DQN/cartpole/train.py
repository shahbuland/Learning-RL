import gym
from agent import Agent
from constants import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make("CartPole-v1")
agent = Agent()
time_limit = 500

GRAPH = [[],[],[]] # episode / reward/ loss
won = False

def add_to_graph(e,r,l):
	global GRAPH
	if(len(GRAPH[0]) > 100):
		del GRAPH[0][0]
		del GRAPH[1][0]
		del GRAPH[2][0]
	GRAPH[0].append(e)	
	GRAPH[1].append(r)
	GRAPH[2].append(l)

plt.ion
plt.figure()
plt.show(block=False)

def draw_graph():
	x = np.asarray(GRAPH[0])
	r = np.asarray(GRAPH[1])
	l = np.asarray(GRAPH[2])
	plt.pause(0.0001)
	plt.cla()
	plt.plot(x,r, color='red')
	plt.plot(x,l, color='blue')
	plt.draw()

for e in range(EPISODES):
	total_r = 0
	s = env.reset()

	for t in range(time_limit):
		if won: env.render()
		a = agent.act(s)
		s_new, r, done, _ = env.step(a)
		total_r += r
		agent.add_exp([s, a, r, s_new, done])
		s = s_new
		if done: break
	if total_r > SCORE_MAX:
		print("Agent won!")
		won = True
	if not won:
		q_loss = agent.replay(32)
		print("Episode", e, "| Reward:", total_r, "| Loss:", q_loss)
		add_to_graph(e,total_r,q_loss)
		draw_graph()
