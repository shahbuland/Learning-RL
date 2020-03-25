import torch

# transitions assumed to be of form:
# state, action, reward, next_state, 
def train_on_transitions(transitions, model, opt, epochs, batch_size)
	size = len(transitions)
	for epoch in range(epochs):
		 
