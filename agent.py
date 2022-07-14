import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from game import SnakeGame
from model import Net
from memory import ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import sleep
import pygame
import torchvision.transforms as T
from plot import plot
import os

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))


N_ACTIONS = 3
REPLAY_MEMORY_SIZE = 100000
LEARNING_RATE = .003

class Linear_QNet(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		return x

	def save(self, file_name='model.pth'):
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)

class QTrainer:
	def __init__(self, model, lr, gamma):
		self.lr = lr
		self.gamma = gamma
		self.model = model
		self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()

	def train_step(self, state, action, reward, next_state, done):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)
		# (n, x)

		if len(state.shape) == 1:
			# (1, x)
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			done = (done, )

		# 1: predicted Q values with current state
		pred = self.model(state)

		target = pred.clone()
		for idx in range(len(done)):
			Q_new = reward[idx]
			if not done[idx]:
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

			target[idx][torch.argmax(action[idx]).item()] = Q_new
	
		# 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
		# pred.clone()
		# preds[argmax(action)] = Q_new
		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward()

		self.optimizer.step()

class DQN(nn.Module):

	def __init__(self, h, w, outputs):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = x.to(device)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 20

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape
HEIGHT = 30
WIDTH = 48
# HEIGHT = 30
# WIDTH = 48

# Get number of actions from gym action space
n_actions = 3

policy_net = DQN(HEIGHT, WIDTH, n_actions).to(device)
target_net = DQN(HEIGHT, WIDTH, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			# print(policy_net(state).argmax())
			return policy_net(state).argmax()
			return policy_net(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
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
	# print(transitions)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))
	for i, m in enumerate(batch.state):
		if m == None:
			print(i)
	# print(batch.state)

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

# class Agent:

# 	def __init__(self):
# 		self.n_games = 0
# 		self.epsilon = 0 # randomness
# 		self.gamma = 0.9 # discount rate
# 		self.memory = deque(maxlen=REPLAY_MEMORY_SIZE) # popleft()
# 		# self.model = DQN()
# 		self.model = Linear_QNet(11, 256, 3)
# 		self.trainer = QTrainer(self.model, lr=.001, gamma=self.gamma)


# 	def get_state(self, game: SnakeGame):
# 		# head = game.snake[0]
# 		# point_l = Point(head.x - 20, head.y)
# 		# point_r = Point(head.x + 20, head.y)
# 		# point_u = Point(head.x, head.y - 20)
# 		# point_d = Point(head.x, head.y + 20)
		
# 		# dir_l = game.direction == Direction.LEFT
# 		# dir_r = game.direction == Direction.RIGHT
# 		# dir_u = game.direction == Direction.UP
# 		# dir_d = game.direction == Direction.DOWN

# 		# state = [
# 		# 	# Danger straight
# 		# 	(dir_r and game.is_collision(point_r)) or 
# 		# 	(dir_l and game.is_collision(point_l)) or 
# 		# 	(dir_u and game.is_collision(point_u)) or 
# 		# 	(dir_d and game.is_collision(point_d)),

# 		# 	# Danger right
# 		# 	(dir_u and game.is_collision(point_r)) or 
# 		# 	(dir_d and game.is_collision(point_l)) or 
# 		# 	(dir_l and game.is_collision(point_u)) or 
# 		# 	(dir_r and game.is_collision(point_d)),

# 		# 	# Danger left
# 		# 	(dir_d and game.is_collision(point_r)) or 
# 		# 	(dir_u and game.is_collision(point_l)) or 
# 		# 	(dir_r and game.is_collision(point_u)) or 
# 		# 	(dir_l and game.is_collision(point_d)),
			
# 		# 	# Move direction
# 		# 	dir_l,
# 		# 	dir_r,
# 		# 	dir_u,
# 		# 	dir_d,
			
# 		# 	# Food location 
# 		# 	game.food.x < game.head.x,  # food left
# 		# 	game.food.x > game.head.x,  # food right
# 		# 	game.food.y < game.head.y,  # food up
# 		# 	game.food.y > game.head.y  # food down
# 		# 	]
# 		print(game.get_env())
# 		return game.get_env()
# 		# return np.array(state, dtype=int)

# 	def remember(self, state, action, reward, next_state, done):
# 		self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

# 	def train_long_memory(self):
# 		if len(self.memory) > BATCH_SIZE:
# 			mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
# 		else:
# 			mini_sample = self.memory

# 		states, actions, rewards, next_states, dones = zip(*mini_sample)
# 		self.trainer.train_step(states, actions, rewards, next_states, dones)
# 		#for state, action, reward, nexrt_state, done in mini_sample:
# 		#	self.trainer.train_step(state, action, reward, next_state, done)

# 	def train_short_memory(self, state, action, reward, next_state, done):
# 		self.trainer.train_step(state, action, reward, next_state, done)

# 	def get_action(self, state):
# 		# random moves: tradeoff exploration / exploitation
# 		self.epsilon = 80 - self.n_games
# 		final_move = [0,0,0]
# 		if random.randint(0, 200) < self.epsilon:
# 			move = random.randint(0, 2)
# 			final_move[move] = 1
# 		else:
# 			state0 = torch.tensor(state, dtype=torch.float)
# 			prediction = self.model(state0)
# 			move = torch.argmax(prediction).item()
# 			final_move[move] = 1

# 		return final_move


# def train():
# 	plot_scores = []
# 	plot_mean_scores = []
# 	total_score = 0
# 	record = 0
# 	pygame.init()
# 	font = pygame.font.SysFont('arial', 25)
# 	agent = Agent()
# 	game = SnakeGame(w=WIDTH, h=HEIGHT, font=font)
# 	while True:
# 		# get old state
# 		state_old = agent.get_state(game)

# 		# get move
# 		final_move = agent.get_action(state_old)
# 		# print(final_move)

# 		# perform move and get new state
# 		score, reward, done = game.play_step(final_move)
# 		print(reward)
# 		state_new = agent.get_state(game)

# 		# train short memory
# 		agent.train_short_memory(state_old, final_move, reward, state_new, done)

# 		# remember
# 		agent.remember(state_old, final_move, reward, state_new, done)

# 		if done:
# 			# train long memory, plot result
# 			game.reset()
# 			agent.n_games += 1
# 			agent.train_long_memory()

# 			if score > record:
# 				record = score
# 				agent.model.save()

# 			print('Game', agent.n_games, 'Score', score, 'Record:', record)

# 			plot_scores.append(score)
# 			total_score += score
# 			mean_score = total_score / agent.n_games
# 			plot_mean_scores.append(mean_score)
# 			plot(plot_scores, plot_mean_scores)
# 	pygame.quit()

def main():
	if SnakeGame.NORMAL_GAME:
		pygame.init()
		font = pygame.font.SysFont('arial', 25)
		game = SnakeGame(w=WIDTH, h=HEIGHT, font=font)
		sleep(2)
		game_over = False
		while not game_over:
			game_over, score, reward, state = game.play_step()
		print('Final Score', score)
		pygame.quit()
	else:
		if SnakeGame.DISPLAY:
			pygame.init()
			font = pygame.font.SysFont('arial', 25)
		else:
			font = None
		env = SnakeGame(font=font)
		game_over = False

		# while not game_over:
		# 	# Choose random action
		# 	action = np.zeros(3)
		# 	action[np.random.randint(0, 3)] = 1
		# 	print(action, game.snake)
	
		# 	state = game.get_state(verbose=True)
		# 	game.play_step(action=action)
		# 	state = game.get_state(verbose=True)
		# 	game_over = state[2]
		# 	score = state[0]
		# 	sleep(.4)

		num_episodes = 1500
		n_games = 0
		plot_scores = []
		plot_mean_scores = []
		total_score = 0
		record = 0

		for i_episode in range(num_episodes):
			# Initialize the environment and state
			env.reset()
			# last_screen = env.get_env()
			current_screen = env.get_env()
			state = current_screen
			# state = current_screen - last_screen
			score = 0
			for t in count():
				# env.get_state(verbose=True)
				# Select and perform an action
				action = select_action(state)
				action_game = np.zeros(3)
				action_game[action.item()] = 1
				# print("Action:", action, action_game)
				score, reward, done = env.play_step(action=action_game)
				# if t % 10 == 0:
				print(reward, score)
				env.get_state(verbose=True)
				reward = torch.tensor([reward], device=device)

				# Observe new state
				# last_screen = current_screen
				current_screen = env.get_env()
				if not done:
					next_state = current_screen
					# next_state = current_screen - last_screen
				else:
					next_state = None

				# Store the transition in memory
				print(state, action, next_state, reward)
				memory.push(state, action, next_state, reward)

				# Move to the next state
				state = next_state

				# Perform one step of the optimization (on the policy network)
				optimize_model()
				if done:
					env.reset()
					episode_durations.append(score)
					n_games += 1

					if score > record:
						record = score

					print('Game', n_games, 'Score', score, 'Record:', record)

					plot_scores.append(score)
					total_score += score
					mean_score = total_score / n_games
					plot_mean_scores.append(mean_score)
					plot(plot_scores, plot_mean_scores)
					# plot_durations()
					break
			# Update the target network, copying all weights and biases in DQN
			# print(score)
			if i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(policy_net.state_dict())

		print('Complete')
		# env.render()
		# env.close()
		plt.ioff()
		plt.show()

		print('Final Score', score)
		if SnakeGame.DISPLAY:
			pygame.quit()


if __name__ == '__main__':
	main()
