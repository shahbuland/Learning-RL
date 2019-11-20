import gym
from agent import Agent
from constants import *
import torch
import numpy as np
import time
from ops import prep_state
import matplotlib.pyplot as plt

env = gym.make(ENV_NAME)
agent = Agent()

GRAPH = [[],[],[], []] # episode / reward/ avg reward/ loss
won = False

def add_to_graph(e,r,ar,l):
	global GRAPH
	if(len(GRAPH[0]) > 100):
		del GRAPH[0][0]
		del GRAPH[1][0]
		del GRAPH[2][0]
		del GRAPH[3][0]
	GRAPH[0].append(e)	
	GRAPH[1].append(r)
	GRAPH[2].append(ar)
	GRAPH[3].append(l)

plt.ion
plt.figure()
plt.show(block=False)

def draw_graph():
	x = np.asarray(GRAPH[0])
	r = np.asarray(GRAPH[1])
	ar = np.asarray(GRAPH[2])
	l = np.asarray(GRAPH[3])
	plt.pause(0.0001)
	plt.cla()
	plt.plot(x,r, color='red')
	plt.plot(x,ar,color='orange')
	plt.plot(x,l, color='blue')
	plt.draw()

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
	q_loss = 0
	lives = 5
	for t in range(TIME_LIMIT):

		env.render()

		a = agent.act(s)

		s_new, r, done, info = env.step(a)
		
		# Punish agent if it loses a life
		new_lives = info['ale.lives']
		
		punishment = 0
		if new_lives < lives:
			punishment = LIFE_LOST_PUNISHMENT * (new_lives - lives)
			lives = new_lives

		# Punishment only passed in replay, total r should measure game score
		# and is primarily just for monitoring the model
		total_r += r
		
		s_new = prep_state(s_new)
		agent.add_exp([s, a, r+punishment, s_new, done])
		
		s = s_new
	
		# Train	
		if not won and (step+1) % TRAINING_INTERVAL == 0:
			q_loss+=agent.replay(BATCH_SIZE)
		if not won and step % TARGET_UPDATE_INTERVAL == 0:
			agent.target.load_state_dict(agent.model.state_dict())
		step += 1
	
		if done: break
	scores.append(total_r)
	# Save weights
	if (e+1) % CHECKPOINT_INTERVAL == 0:
		torch.save(agent.model.state_dict(), "params.pt")
	if total_r > SCORE_MAX:
		print("Agent won!")
		won = True
	if not won:
		print("Episode", e, "| Reward:", total_r, "| Avg Reward:", np.mean(np.asarray(scores)),"| Loss:", np.mean(q_loss))
		add_to_graph(e,total_r,np.mean(np.asarray(scores)), q_loss)	
		draw_graph()	
