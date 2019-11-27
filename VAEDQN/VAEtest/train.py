import gym
from ops import *
from constants import *
import rendering
from vaeagent import Agent
import torch

env = gym.make(env_name)

# Rendering stuff
plt.ion()
fig,axs = plt.subplots(2)
graph = rendering.Graph(4,100)

agent = Agent()
agent.load_params()

for e in range(EPISODES):
	max_r = 0
	s = env.reset()
	s = prep_state(s)
	# Init to 0 so graph doesnt bug out
	q_loss = 0 
	v_loss = 0
	lives = 5
	while True:
		a = agent.act(s)
		s_new, r, done, info = env.step(a)
		
		# Punish for lost lives
		new_lives = info['ale.lives']
		punishment = 0
		if new_lives < lives:
			punsihment = LIFE_LOST_PUNISHMENT * (new_lives - lives)
			lives = new_lives

		# Punishment only passed in replay, total r is for monitoring
		max_r = max(r,max_r)
		s_new = prep_state(s_new)
		agent.add_exp([s,a,r+punishment,s_new,done])
		s = s_new

		# Training
		v_loss, q_loss = agent.replay(DQN_BATCH_SIZE)
		if done: break
		
		# Save weights
		if (e+1) % SAVE_INTERVAL == 0:
			agent.save_params()
		
		# Add to graph
		graph.add_data([agent.action_number, q_loss, v_loss/1e5,r,max_r])
		# Rendering stuff
		plt.pause(0.0001)
		# Draw rec state
		if s is not None:
			rec_state = agent.vision(s)[2]
			rendering.render_state(axs[0],rec_state)
		# Draw graph
		graph.draw_graph(axs[1])				
		plt.draw()
