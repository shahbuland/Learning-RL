MIN_BUFFER_SIZE = 10000 # Minimum size for experience buffer to start training
MAX_BUFFER_SIZE = 100000 # Maximum size for experience buffer

NUM_ACTIONS = 4 # Number of actions agent could take

GAMMA = 0.99 # Decay for bellman equation
EPSILON = 1 # Initial chance to act randomly
EPSILON_DECAY = 0.8 # Decay for epsilon
EPSILON_MIN = 0.05 # Min val for epsilon
LEARNING_RATE = 0.01

BATCH_SIZE = 4
TRAINING_INTERVAL = 4 # Step interval for training
TARGET_UPDATE_INTERVAL = 1000 # Step interval for updating target

SCORE_MAX = 20 # Score before agent 'wins'
EPISODES = 1000 # Episodes to train for
TIME_LIMIT = 18000
LIFE_LOST_PUNISHMENT = 1 # Punishment for losing life
CHECKPOINT_INTERVAL = 10 # Interval on which to take checkpoints
LOAD_CHECKPOINTS = True

ENV_NAME = 'BreakoutNoFrameskip-v4'
USE_CUDA = False
WEIGHT_CLIPPING = True

