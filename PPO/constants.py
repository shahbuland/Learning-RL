
# Model I/O
CHANNELS = 3
IMAGE_SIZE = 84 # Width/Height of frame when input to model
OUTPUT_SIZE = 4
POLICY_STD = 0 # Actor output is distribution, what should std be?

# Game
ENV_NAME = "BreakoutNoFrameskip-v4"
EPISODES = 1000

# Training
USE_CUDA = True
EPOCHS_PER_TRAIN = 3
BATCH_SIZE = 32
NUM_STEPS = 128 # aka rollout size 
CRITIC_COEFF = 1  # Weight in loss to critic
ENT_COEFF = 0.01 # Weight in loss to entropy
MAX_FRAMES = 5000 # Max frames before manually ending episode
CLIP_VAL = 0.5

# Hyperparameters
LEARNING_RATE = 2e-4
EPSILON = 0.2 # Clip radius for PPO Loss
GAMMA = 0.99 # Discount factor
TAU = 0.95
