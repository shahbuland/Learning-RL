
# Model I/O
CHANNELS = 3
IMAGE_SIZE = 84 # Width/Height of frame when input to model
OUTPUT_SIZE = 4

# Game
ENV_NAME = "BreakoutNoFrameskip-v4"
EPISODES = 1000

# Training
USE_CUDA = True
EPOCHS_PER_TRAIN = 10
BATCH_SIZE = 16
NUM_STEPS = 500

# Hyperparameters
LEARNING_RATE = 2e-4
EPSILON = 0.2 # Clip radius for PPO Loss
GAMMA = 0.99 # Discount factor

