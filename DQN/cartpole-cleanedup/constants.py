MIN_BUFFER_SIZE = 1000 # Minimum size for experience buffer to start training
MAX_BUFFER_SIZE = 10000 # Maximum size for experience buffer

STATE_SIZE = 2 # Size of state
NUM_ACTIONS = 3 # Number of actions agent could take

GAMMA = 0.8 # Decay for bellman equation
EPSILON = 0.9 # Initial chance to act randomly
EPSILON_DECAY = 0.96 # Decay for epsilon
EPSILON_MIN = 0.05 # Min val for epsilon
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
TRAINING_INTERVAL = 1 # Step interval for training
TARGET_UPDATE_INTERVAL = 10 # Step interval for updating target

SCORE_MAX = 500 # Score before agent 'wins'
EPISODES = 2000 # Episodes to train for
TIME_LIMIT = 500

CHECKPOINT_INTERVAL = 10 # Interval on which to take checkpoints
LOAD_CHECKPOINTS = False

ENV_NAME = 'MountainCar-v0'
USE_CUDA = True

WEIGHT_CLIPPING = False
