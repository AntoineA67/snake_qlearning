import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from time import sleep
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from game import SnakeGame

import pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)
env = SnakeGame(font=font, w=4, h=4, display=True)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.memory = deque([],maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		# self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
		# self.bn2 = nn.BatchNorm2d(32)
		# self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
		# self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 2, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		# convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		# convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		convw = conv2d_size_out(w)
		convh = conv2d_size_out(h)
		linear_input_size = convw * convh * 16
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = x.to(device)
		x = F.relu(self.bn1(self.conv1(x)))
		# x = F.relu(self.bn2(self.conv2(x)))
		# x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

# resize = T.Compose([T.ToPILImage(),
# 					T.Resize(40, interpolation=Image.CUBIC),
# 					T.ToTensor()])


# def get_cart_location(screen_width):
# 	world_width = env.x_threshold * 2
# 	scale = screen_width / world_width
# 	return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
	# Returned screen requested by gym is 400x600x3, but is sometimes larger
	# such as 800x1200x3. Transpose it into torch order (CHW).
	screen = env.get_env().transpose((2, 0, 1))
	# print(screen)
	screen = np.ascontiguousarray(screen, dtype=np.float32)
	screen = torch.from_numpy(screen)
	# screen = torch.nn.Upsample(scale_factor=2, mode='nearest')(screen)
	# Resize, and add a batch dimension (BCHW)
	# print(screen.unsqueeze(0).shape, screen.unsqueeze(0))
	
	return screen.unsqueeze(0)


# env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
# 		   interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
print(init_screen.shape)

# Get number of actions from gym action space
n_actions = 3

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
	global steps_done
	sample = random.random()
	if steps_done > 3000:
		eps_threshold = EPS_END + (EPS_START - EPS_END) * \
			math.exp(-1. * steps_done / EPS_DECAY)
	else:
		eps_threshold = .9
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			return policy_net(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
eps = []
record_arr = []

def plot_durations(record):
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.plot(durations_t.numpy())
	if steps_done > 3000:
		eps_threshold = EPS_END + (EPS_START - EPS_END) * \
			math.exp(-1. * steps_done / EPS_DECAY)
	else:
		eps_threshold = .9
	eps.append(eps_threshold)
	record_arr.append(record)
	plt.plot(eps)
	plt.plot(record_arr)
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()


def main():
	num_episodes = 5000
	plt.figure()
	record = 0
	_, ax = plt.subplots(ncols=2)
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		env.reset()
		last_screen = get_screen()
		current_screen = get_screen()
		state = current_screen
		# state = current_screen * last_screen

		for t in count():
			# Select and perform an action
			# print(state[0, 0].numpy())
			# print(state[0, 1].numpy())
			# print(state[0, 2].numpy())
			action = select_action(state)
			# action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

			# print(action.item())
			score, reward, done = env.play_step(action.item())
			reward = torch.tensor([reward], device=device)

			# Observe new state
			last_screen = current_screen
			current_screen = get_screen()
			if not done:
				next_state = current_screen
				# next_state = current_screen * last_screen
			else:
				next_state = None

			# Store the transition in memory
			if score > record:
				print(score)
				record = score
			memory.push(state, action, next_state, reward)

			# Move to the next state
			# if next_state is not None:
			# 	if last_screen is not None:
			# 		ax[0].imshow(last_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
			# 				interpolation='none')
			# 	ax[1].imshow(next_state.cpu().squeeze(0).permute(1, 2, 0).numpy(),
			# 			interpolation='none')
			# 	plt.title(f'AI vision, reward: {reward.item()}')
			# 	plt.show()
			# state = next_state
			# input()
			# sleep(.2)

			# Perform one step of the optimization (on the policy network)
			optimize_model()
			if done:
				episode_durations.append(score)
				plot_durations(record)
				break
		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())

	print('Complete')
	env.render()
	env.close()
	plt.ioff()
	plt.show()

if __name__ == '__main__':
	main()