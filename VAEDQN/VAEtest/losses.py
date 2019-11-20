import torch
from torch import nn
import torch.nn.functional as F

# y = [mu, logvar, rec_x]
# where x is the original input
def VAE_LOSS(y, x):
	mu, logvar, rec_x = y

	# Reconstrution loss (enforces the autoencoder part)
	rec_loss = F.binary_cross_entropy(rec_x, x)

	# KL divergence loss (enforces continuity in latent distribution) (enforces variatonal part)
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	return rec_loss + kl_loss
