# Env info
ENV_NAME = "Cartpole-v0"
ACTION_DIM = 4
NET_INPUT = 4 # State size if its not an image
USE_CONV = False # If state is image

# Model I/O
IMAGE_SIZE = 84 # Size for input to conv layers
CHANNELS = 1

# Training
LEARNING_RATE = 2e-3
BETAS = (0.5,0.999)
GAMMA = 0.99 # Discount factor
TAU = 1
USE_CUDA = False # GPU training
UPDATES = 10000
