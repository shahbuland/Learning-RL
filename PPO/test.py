from train import train_on_env
from models import ActorCritic
from constants import *

model = ActorCritic(use_conv=False,input_size=4)
if USE_CUDA: model.cuda()

train_on_env("CartPole-v0", model, 5000, 128)
