import gym

from torch import nn
from torch.distributions.categorical import Categorical 
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from stable_baselines.common.vec_env import DummyVecEnv

from constants import *
import ops

# Trains over data provided
# Must provide, model, optimizer, states, rewards, actions taken, 
# advantages (critic output), resultant states and action probabilities (actor outputs)
def train_on_batch(model, optimizer, state_batch, reward_batch, actions_batch,
	adv_batch, state_new_batch, old_act_probs):

	size = len(state_batch)
	loss = nn.MSELoss()

	# Convert lists of old_act_probs from all workers 
	# Into a single list of act probs
	old_act_prob_list = torch.stack(old_act_probs).permute(1, 0, 2).view(-1, OUTPUT_SIZE)
	if USE_CUDA: old_act_prob_list = old_act_prob_list.cuda()
	# Sample a distribution of actions from old probabilities
	# i.e. we get the old action probability distribution
	model_dist_old = Categorical(action_probs_old_list)
	# Get probabilities of the actions that were taken
	# With respect to the old model
	log_prob_old = model_dist_old.log_prob(actions_batch)

	for EPOCH in range(EPOCHS_PER_TRAIN):
		# Get indices
		ind = torch.randint(0, size, (BATCH_SIZE,))

		# Current models act probs and values
		new_act_probs, new_value = model(state_batch[ind])
		# Get distribution for actions and then the probabilities of taking actions batch
		model_dist_new = Categorical(new_act_probs)
		log_prob_new = model_dist_new.log_prob(actions_batch[ind])

		# Now we begin calculating PPO Loss
		ratio = (log_prob_new - log_prob_old[ind]).exp()
		# Left and right expressions in PPO Loss function
		left_exp = ratio * adv_batch[ind]
		right_exp = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * adv_batch[ind]
		
		# Losses for actor (policy) and critic 
		actor_loss = -torch.min(left_exp, right_exp).mean()
		critic_loss = F.mse_loss(new_value.sum(1), reward_batch[ind])

		# Entropy from models action distribution
		entropy = model_dist_new.entropy().mean()

		optimizer.zero_grad()
		loss = actor_loss + critic_loss
		loss.backward()
		optimizer.step()

# Train model on specified environment for given number of episodes 
# with given number of workers
# Verbose determines if we should log (to tensorboard)
# Render determines if we should show env while training
def train_on_vec_env(model, env_name, num_workers, episodes, verbose = True, render = True):

	# Baselines vec env is kind of weird and required functions as inputs for environments
	# Hence the existence of the function below
	def get_env():
		return gym.make(env_name)

	env = DummyVecEnv([get_env for _ in range(num_workers)]

	opt = torch.optim.Adam(model.parameters(lr= LEARNING_RATE))

	print("Training")
	for EPISODE in range(episodes):
		states = []
		rewards = []
		dones = []
		next_states = []
		actions = []
		values = []
		action_probs = []

		# Reseting env gives us num_workers obs
		s = env.reset()
		s = ops.prep_obs(s)

		# Roll out
		for STEP in range(NUM_STEPS):
			# Get act probs from multiple workers
			act_prob, val = model(s)
			act = [ops.sample_from(act_prob[i] for i in range(num_workers)]

			env.step_async(act)
			s_next, r, _, done = step_wait()
			s_next = ops.prep_obs(s)  		
	
	
# Train in single environment
# Otherwise arguments same as above
def train_on_env(mode, env, episodes, verbose = True, render = True):
	env = gym.make(env_name)
	
	
