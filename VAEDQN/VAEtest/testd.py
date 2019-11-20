from constants import *
from models import Decoder
import torch

x = torch.ones(1,LATENT_DIM)
d = Decoder()
y = d(x)
print(y.shape)

