MIN_BUFFER_SIZE = 1000 # Minimum size for experience buffer to start training
MAX_BUFFER_SIZE = 10000 # Maximum size for experience buffer

NUM_ACTIONS = 4 # Number of actions agent could take

GAMMA = 0.99 # Decay for bellman equation
EPSILON = 1 # Initial chance to act randomly
EPSILON_DECAY = 0.999 # Decay for epsilon
EPSILON_MIN = 0.02 # Min val for epsilon
LEARNING_RATE = 0.0025

BATCH_SIZE = 32
TRAINING_INTERVAL = 4 # Step interval for training
TARGET_UPDATE_INTERVAL = 1000 # Step interval for updating target

SCORE_MAX = 20 # Score before agent 'wins'
EPISODES = 1000 # Episodes to train for
TIME_LIMIT = 25000

CHECKPOINT_INTERVAL = 10 # Interval on which to take checkpoints
LOAD_CHECKPOINTS = True

ENV_NAME = 'Pong-v0'
USE_CUDA = True
WEIGHT_CLIPPING = True
