# Game specific:
STATE_SIZE = 64*6*6 # Linear dim of state (what feature maps become)
ACTION_SIZE = 6 # Number of actions we can take
ENV_NAME = 'QbertNoFrameskip-v4'
TIME_LIMIT = 5000
# For cropping in state pre processing
CROP = False
TOP_BORDER = 0 # How far to crop from top
BOTTOM_BORDER = 0 # How far from bottom

# Training:
USE_CUDA = True
CLIP_WEIGHTS = False
EPISODES = 1000
UPDATE_INTERVAL = 5 # Steps between updates
SAVE_INTERVAL = 500 # Interval on which to save parameters
LOAD_CHECKPOINTS = False

# Model hyperparameters:
LEARNING_RATE = 1e-7
GAMMA = 0.99
ENT_WEIGHT = 0.01
GRAD_MAX_NORM = 0.5
