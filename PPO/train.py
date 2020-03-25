import gym

import torch
from torch import nn
from torch.distributions.categorical import Categorical 
import torch.nn.functional as F

from tensorboardX import SummaryWriter

#from stable_baselines.common.vec_env import DummyVecEnv

from constants import *
import ops
from rollout_storage import RolloutStore

# Trains over data provided
# Must provide a valid rollout store (we assume all data same length)
def train_PPO(model, optimizer, rollout_store):

	old_log_probs, val_batch, state_batch, actions_batch, returns_batch, _ = rollout_store.unwind()
	
	# Detach a few things from autograd to prevent bugs
	returns_batch = returns_batch.detach()
	old_log_probs = old_log_probs.detach()
	val_batch = val_batch.detach()
	
	# Get advantages from returns and values
	adv_batch = returns_batch - val_batch
	# Normalize
	adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std()) + 1e-8

	size = len(state_batch)
	total_loss = 0
	
	for EPOCH in range(EPOCHS_PER_TRAIN):
		# Shuffle indices
		full_ind = torch.randint(0, size, (size,))
		
		# mini batch training over all data
		# This is worded kind of weirdly but it iterates through the above
		# indices in [BATCH_SIZE] sized chunks
		for ind in [full_ind[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(size // BATCH_SIZE)]:
			# Current models distribution and values	
			new_dist, new_value = model(state_batch[ind])
			# Get new probs and entropy 
			new_log_probs = new_dist.log_prob(actions_batch[ind])

			# Now we begin calculating PPO Loss
			# Really, we are taking a ratio (pi_new / pi_old)
			# The reason all this log stuff is happening is because
			# For small numbers, log protects from rounding errors,
			# and turns log(a/b) into log(a) - log(b)
			ratio = (new_log_probs - old_log_probs[ind]).exp()
			# Left and right expressions in PPO Loss function
			left_exp = ratio * adv_batch[ind]
			right_exp = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * adv_batch[ind]
			
			# Losses for actor (policy) and critic
			# critic loss (value loss) is simple MSE between observed
			# and predicted reward
			# actor loss is * -1  as the internal expression is intended
			# to be maximized (it is a prediction of policy reward)
			# Additionally, the minimum is taken so that if the ratio
			# (difference) between old and new policy is too large, we
			# don't take too big a step (we are uncertain of whether new
			# policy is really better or not)
			actor_loss = -torch.min(left_exp, right_exp).mean()
			critic_loss = (new_value - returns_batch[ind]).pow(2).mean()

			# Entropy from models action distribution
			entropy = new_dist.entropy().mean()
			
			loss = actor_loss + CRITIC_COEFF * critic_loss - ENT_COEFF * entropy
			optimizer.zero_grad()
			loss.backward()
			# gradient clipping
			nn.utils.clip_grad_norm_(model.parameters(), CLIP_VAL)
			optimizer.step()
			total_loss += loss.item()

		return total_loss

# Train ppo on environment provided
def train_on_env(env_name, model, episodes, rollout_length):
	# create env first
	env = gym.make(env_name)
	
	RS = RolloutStore()
	opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
	TOTAL_FRAMES = 0
	for EPISODE in range(episodes):
		s = ops.prep_obs(env.reset())
		done = False
		frame = 0 # frame we are on 
		total_r = 0 # Worth tracking

		while frame < MAX_FRAMES and not done:
			RS.reset()
			entropy = 0

			for STEP in range(NUM_STEPS):
				env.render()
				# Get distribtion, value and sample action
				dist, val = model(s)
				a = dist.sample()
				# step with action, get next state, reward
				s_next, r, done, _ = env.step(a.cpu().item())
				# Preprocess state and add reward
				s_next = ops.prep_obs(s_next)
				total_r += r
					
				# Log prob and entropy
				log_p = dist.log_prob(a)
				entropy += dist.entropy().mean()

				# Make mask
				done = 1 if done else 0
				m = ops.makeMask(done)
				
				# For below, some notes:
				# advantage is given by r - val (hence why we store it)
				# log probs are used because they scale better for learning
				# r is updated so that it is a valid tensor when put in list
				r = torch.FloatTensor([r]).unsqueeze(1)
				if USE_CUDA: r = r.cuda()
				RS.add([log_p, val, s, a, r, m]) 
				
				# Update frame count, state and end early if done
				frame += 1
				s = s_next

			# Update rollout store's rewards with returns from GAE
			_, next_v = model(s_next)
			RS.update_gae(next_v)	
			loss = train_PPO(model, opt, RS)
			print(loss)
		if EPISODE % 10 == 0:
			model.save_weights("firstrun.pt")
		TOTAL_FRAMES+=NUM_STEPS	
		print("Steps", TOTAL_FRAMES, " | Reward:", total_r)
