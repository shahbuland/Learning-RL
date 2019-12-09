# Game specific:
STATE_SIZE = 512*5*5 # Linear dim of state (what feature maps become)
ACTION_SIZE = 4 # Number of actions we can take
ENV_NAME = 'BreakoutNoFrameskip-v4'
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
