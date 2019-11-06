MIN_BUFFER_SIZE = 100 # Minimum size for experience buffer to start training
MAX_BUFFER_SIZE = 1000 # Maximum size for experience buffer

STATE_SIZE = 4 # Size of state
NUM_ACTIONS = 2 # Number of actions agent could take

GAMMA = 0.9 # Decay for bellman equation
EPSILON = 1 # Initial chance to act randomly
EPSILON_DECAY = 0.99 # Decay for epsilon
EPSILON_MIN = 0.05 # Min val for epsilon
LEARNING_RATE = 0.01

EPISODES = 1000 # Episodes to train for
