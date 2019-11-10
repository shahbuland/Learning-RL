MIN_BUFFER_SIZE = 1000 # Minimum size for experience buffer to start training
MAX_BUFFER_SIZE = 10000 # Maximum size for experience buffer

STATE_SIZE = 2
NUM_ACTIONS = 3 # Number of actions agent could take

GAMMA = 0.95 # Decay for bellman equation
EPSILON = 1 # Initial chance to act randomly
EPSILON_DECAY = 0.995 # Decay for epsilon
EPSILON_MIN = 0.01 # Min val for epsilon
LEARNING_RATE = 0.001

TRAINING_INTERVAL = 200 # Step interval foor training

EPISODES = 1000 # Episodes to train for
TIME_LIMIT = 9999

CHECKPOINT_INTERVAL = 3 # Interval on which to take checkpoints
LOAD_CHECKPOINTS = False

ENV_NAME = 'MountainCar-v0'
