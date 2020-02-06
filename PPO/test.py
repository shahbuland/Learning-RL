from train import train_on_env
from models import ActorCritic
from constants import *

model = ActorCritic()
if USE_CUDA: model.cuda()

train_on_env("BreakoutNoFrameskip-v4", model, 1000, 100)
