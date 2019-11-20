from models import VAE
import gym
from ops import *
from constants import *
import random
from losses import VAE_LOSS

model = VAE().cuda() if USE_CUDA else VAE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

env = gym.make('Pong-v0')

plt.ion()
plt.figure()
plt.show(block = False)

def render_state(s_rec):
	plt.pause(0.0001)
	plt.cla()
	s_rec = s_rec.cpu().detach().numpy()
	s_rec = np.squeeze(s_rec)
	plt.imshow(s_rec,cmap='gray')
	plt.draw()

state_memory = []
	
MAX_SIZE = 10000
MIN_SIZE = 100
BATCH_SIZE = 32

while True:
	s = env.reset()
	while True:
		s, _, _, _ = env.step(random.randint(0,2))
		s = prep_state(s)
		if len(state_memory) >= MAX_SIZE:
			del state_memory[0]

		state_memory.append(s)

		
		if len(state_memory) >= MIN_SIZE:
			
			# Viewing
			s_rec = model.decode(model.encode(s))
			env.render()
			render_state(s_rec)
			
			# Train vae
			opt.zero_grad()
			batch = random.sample(state_memory, BATCH_SIZE)
			s_batch = torch.cat(batch)
			output_batch = model(s_batch) # [mu, logvar, rec_x]
			loss = VAE_LOSS(output_batch, s_batch)
			loss.backward()
			opt.step()
			print(loss.item())  
				
