# Purpose of this folder:

DQNs are outdated but still a good way to start thinking about RL. 
They work by feeding game state into a network and predicting quality
of any given action, then predicting the quality of that action
in the next state. It chooses an action that maximizes reward immediately  
and in the future. Doesn't work too well on breakout (probably due to choice of
network architecture and hyperparameters).
