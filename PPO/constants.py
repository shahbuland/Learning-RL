
# Model I/O
CHANNELS = 1
INPUT_IMAGE = False # Is input image?
IMAGE_SIZE = 84 # Width/Height of frame when input to model
OUTPUT_SIZE = 2
POLICY_STD = 0 # Actor output is distribution, what should std be?

# Game
ENV_NAME = "BreakoutNoFrameskip-v4"
EPISODES = 1000

# Training
USE_CUDA = False
EPOCHS_PER_TRAIN = 4
BATCH_SIZE = 4
NUM_STEPS = 128 # aka rollout size 
CRITIC_COEFF =  1  # Weight in loss to critic
ENT_COEFF = 0.01 # Weight in loss to entropy
MAX_FRAMES = 5000 # Max frames before manually ending episode
CLIP_VAL = 0.5

# Hyperparameters
LEARNING_RATE = 3e-4
EPSILON = 0.2 # Clip radius for PPO Loss
GAMMA = 0.99 # Discount factor
TAU = 0.95
