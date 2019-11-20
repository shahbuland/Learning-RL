import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from constants import *

# Assuming state size is 84
class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		
		# Conv layers
		self.conv1 = nn.Conv2d(1, 64, 3, 1, 1) # 41
		self.conv2 = nn.Conv2d(64, 128, 3, 1, 1) # 20
		self.conv3 = nn.Conv2d(128,256, 3, 1, 1) # 10
		self.conv4 = nn.Conv2d(256,512, 3, 1, 1) # 5 (interpolate to 4)
		# Size becomes 256 x 4 x 4

		# Batch norm layers
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(512)
		
		# fc layer
		self.fc1 = nn.Linear(512*4*4,1024)
		
		self.mu_fc = nn.Linear(1024, LATENT_DIM)
		self.logvar_fc = nn.Linear(1024, LATENT_DIM)

	def sample(self, mu, logvar):
		# We want a number sampled from the LATENT_DIM dimensional gaussian
		# With a mean at vector mu and sigmas given by logvar
		std = logvar.mul(0.5).exp_()
		Z = torch.randn_like(std) # Z ~ G(0,1)
		enc = mu + Z*std # enc ~ G(mu, std)
		return enc

	def forward(self, x):
		x = self.conv1(x)
		x = nn.MaxPool2d((2,2))(x)
		x = F.relu(x)
		x = self.bn1(x)
		x = self.conv2(x)
		x = nn.MaxPool2d((2,2))(x)
		x = F.relu(x)
		x = self.bn2(x)
		x = self.conv3(x)
		x = nn.MaxPool2d((2,2))(x)
		x = F.relu(x)
		x = self.bn3(x)
		x = self.conv4(x)
		x = nn.MaxPool2d((2,2))(x)
		x = F.relu(x)
		x = self.bn4(x)
		
		x = F.interpolate(x, (4,4))
		x = x.view(-1,512*4*4)
		x = self.fc1(x)
		x = F.relu(x)
	
		mu = self.mu_fc(x)
		logvar = self.logvar_fc(x)

		z = self.sample(mu, logvar)
		
		return mu, logvar, z

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		
		self.fc1 = nn.Linear(LATENT_DIM, 1024)
		self.fc2 = nn.Linear(1024, 512*4*4)

		self.conv1 = nn.Conv2d(512,256,3,1,1) # -> 10
		self.conv2 = nn.Conv2d(256,128,3,1,1) # -> 20
		self.conv3 = nn.Conv2d(128,64,3,1,1) # -> 40
		self.conv4 = nn.Conv2d(64,1,3,1,1) # -> 84

	def forward(self, x):
		
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		x = x.view(-1,512,4,4)

		x = self.conv1(x)
		x = F.relu(x)
		x = F.interpolate(x, (10,10))
		x = self.conv2(x)
		x = F.relu(x)
		x = F.interpolate(x, (20,20))
		x = self.conv3(x)
		x = F.relu(x)
		x = F.interpolate(x, (40,40))
		x = self.conv4(x)
		x = torch.sigmoid(x)
		
		# Force 1 x 84 x 84 shape
		x = F.interpolate(x, (84,84))

		return x

		
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		
		self.E, self.D = Encoder(), Decoder()
	
	def encode(self, x):
		return self.E(x)[2]
	
	def decode(self,x):
		return self.D(x)

	def forward(self, x):
		mu, logvar, z = self.E(x)
		rec_x = self.D(z)

		return mu, logvar, rec_x
			
