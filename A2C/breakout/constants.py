# Game specific:
STATE_SIZE = 256*5*5 # Linear dim of state (what feature maps become)
ACTION_SIZE = 6 # Number of actions we can take
ENV_NAME = 'PongNoFrameskip-v4'
TIME_LIMIT = 5000

# Training:
USE_CUDA = True
BATCH_SIZE = 32
CLIP_WEIGHTS = True
EPISODES = 1000
UPDATE_INTERVAL = 50 # Steps between updates
SAVE_INTERVAL = 500 # Interval on which to save parameters
LOAD_CHECKPOINTS = True

# Model hyperparameters:
LEARNING_RATE = 1e-3
GAMMA = 0.99
ENT_WEIGHT = 0.001
