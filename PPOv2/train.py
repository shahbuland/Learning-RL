import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import gym

from constants import *
from rollout_store import Rollout_Storage
import ops

# Gets loss given batch of (          
def PPOLoss(model, batch)
# Train using PPO Loss
def PPOtrain(model, opt, store):
	

# Tries model on env for some given number of steps and initial state
# Returns a rollout storage, reward and last state (to pass forward later)
# In: Model, Env (Gym Env), Steps (Int), start_s (State)
# Out: Samples (Rollout storage), Total Reward (int), last_s (State)
# Assumptions: start_s can be none, in which case env is reset
def sample(model, env, steps, start_s):
	store = Rollout_Storage()
	total_r = 0

	s = start_s
	if s is None:
		s = ops.prep_state(env.reset())

	for step in range(steps):
		pi, logits, v = model(s)
		a = pi.sample() # Sample action
		log_p = pi.log_prob(a) # Log prob of "a" being selected
		
		next_s, r, d = ops.mod_step(env, a)
		total_r += r.item()
		store.add([s, next_s, a, log_p, r, d])
		s = next_s

	return store, total_r, s
		

# Trains model on an environment for given number of episodes
def train_on_env(model, env_name, episodes):
	# Initialize an optimizer for model and a summary writer
	opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, betas = BETAS)
	model.train()
	writer = SummaryWriter()

	env = gym.make(env_name)
	next_s = ops.prep_state(env.reset()) # initial state to send to sample

	for update in range(UPDATES):
		samples, total_reward, next_s = sample(model, env, next_s)
		total_loss = PPOtrain(model, opt, samples)
		writer.add_scalar("Loss", total_loss, update)
		writer.add_scalar("Total Reward", total_reward, update)

