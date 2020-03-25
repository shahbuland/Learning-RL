# Game specific:
STATE_SIZE = 4 # Linear dim of state
ACTION_SIZE = 2 # Number of actions we can take
ENV_NAME = 'CartPole-v0'
TIME_LIMIT = 500

# Training:
USE_CUDA = True
BATCH_SIZE = 32
CLIP_WEIGHTS = True
EPISODES = 1000

# Model hyperparameters:
LEARNING_RATE = 3e-4
GAMMA = 0.99
ENT_WEIGHT = 0.001
