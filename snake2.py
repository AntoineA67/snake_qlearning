from keras import layers
import keras
from keras.layers import Dense
from keras.layers.attention.multi_head_attention import activation
from keras.losses import Huber, huber
from game import SnakeGame
import pygame
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import keras as k
import tensorflow as tf
import matplotlib

EPISODES		= 5000
GAMMA			= .99
EPS				= .95
EPS_STEP		= .001
EPS_MIN			= 0.0
BATCH_SIZE		= 100
LEARNING_RATE	= .01
MAX_STEPS		= 200
MAX_MEM_LENGTH	= 100000

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

class Agent:
	def __init__(self) -> None:
		self.model = self.create_model()
		self.model_target = self.create_model()
		self.state_exp = []
		self.reward_exp = []

	def create_model(self):
		inputs = layers.Input(shape=(5,))
		hidden = layers.Dense(50, activation='relu')(inputs)
		hidden2 = layers.Dense(50, activation='relu')(hidden)
		outputs = layers.Dense(3, activation='softmax')(hidden2)
		model = k.Model(inputs=inputs, outputs=[outputs])
		model.compile(optimizer='adam', metrics=['accuracy'])
		return model

	def choose_action(self, state):
		act = self.model(state, training=False)
		# print(act.numpy(), np.argmax(act))
		return np.argmax(act)

	def add_experience(self, state, reward):
		self.state_exp.append(state)
		self.reward_exp.append(reward)

def plot_results(scores, rewards_mean, rand_ratio):
	# plt.figure()
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.plot(scores, label='Score')
	plt.plot(rewards_mean, label='Reward mean')
	# plt.plot(rand_ratio, label='Random ratio')
	plt.legend()

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())

def main():
	pygame.init()
	font = pygame.font.SysFont('Arial', 20)
	game = SnakeGame(font, 10, 10)
	agent = Agent()
	loss_function = Huber()
	scores = []
	rewards = []
	states = []
	next_states = []
	records = [0]
	actions = []
	dones = []
	rewards_mean = []
	rand_ratio = []
	scores_ep = []
	optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
	eps = EPS
	plt.figure()
	
	for ep in range(EPISODES):
		episode_reward = 0
		state = game.get_state()
		r, c = 0, 0
		eps = max(EPS_MIN, eps - EPS_STEP)
		for i in count():
			state_tensor = tf.convert_to_tensor(state)
			state_tensor = tf.expand_dims(state, 0)
			if ep > 50 and np.random.rand(1)[0] > eps:
				action = int(agent.choose_action(state_tensor))
				c += 1
			else:
				r += 1
				action = int(np.random.randint(4))
			score, reward, done = game.play_step(action)
			# next_state = tf.convert_to_tensor(game.get_state())
			# next_state = tf.expand_dims(next_state, 0)
			next_state = np.array(game.get_state())

			episode_reward += reward
			dones.append(done)
			actions.append(action)
			scores.append(score)
			rewards.append(reward)
			states.append(state)
			next_states.append(next_state)
			if reward > records[-1]:
				records.append(reward)
			else:
				records.append(records[-1])
			# agent.add_experience(state, reward)
			state = next_state
			if len(rewards) > MAX_MEM_LENGTH:
				del rewards[:1]
				del states[:1]
				del next_states[:1]
				del actions[:1]
				del dones[:1]
			if done or i > MAX_STEPS:
				break
			# input()

		if len(dones) > BATCH_SIZE:
			indices = np.random.choice(range(len(dones)), size=BATCH_SIZE)

			state_sample = np.array([states[i] for i in indices])
			next_state_sample = np.array([next_states[i] for i in indices])
			action_sample = np.array([actions[i] for i in indices])
			dones_sample = np.array([dones[i] for i in indices])
			reward_sample = np.array([rewards[i] for i in indices])

			future_rewards = agent.model_target.predict(next_state_sample)

			up_q_values = reward_sample + GAMMA * tf.reduce_max(future_rewards, axis=1)

			up_q_values = up_q_values * (1 - dones_sample) - dones_sample

			masks = tf.one_hot(action_sample, 3)

			with tf.GradientTape() as tape:
				q_values = agent.model(state_sample)
				q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
				loss = loss_function(up_q_values, q_action)
			
			grads = tape.gradient(loss, agent.model.trainable_variables)
			optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
			agent.model_target.set_weights(agent.model.get_weights())
			print(f'Episode {ep} Reward {np.mean(rewards)} Random/Chosen {r} {c}')
		rewards_mean.append(np.mean(rewards))
		rand_ratio.append(r / (c + 1e-15))
		scores_ep.append(score)
		plot_results(scores_ep, rewards_mean, rand_ratio)
		game.reset()
	agent.model.save('./model')
	agent.model_target.save('./model_target')

if __name__ == '__main__':
	main()
