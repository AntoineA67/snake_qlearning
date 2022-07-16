from keras.layers import Dense
from game import SnakeGame
import pygame
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import keras as k

EPISODES		= 200
GAMMA			= .99
EPS				= 1.0
EPS_MIN			= .1
EPS_MAX			= 1.0
BATCH_SIZE		= 32
LEARNING_RATE	= .01


class Agent:
	def __init__(self) -> None:
		self.model = self.create_model()
		self.model_target = self.create_model()

	def create_model():
		model = k.Sequential()
		model.add(Dense(50, activation='relu', input_shape=(20,)))
		model.add(Dense(50, activation='softmax'))
		model.compile(learning_rate=LEARNING_RATE, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		return model

	def choose_action(self, state):
		pass

	def train(self, reward: int, done: bool):
		pass

def plot_results():
	pass

def main():
	pygame.init()
	font = pygame.font.SysFont('Arial', 20)
	game = SnakeGame(font, 10, 10)
	agent = Agent()
	scores = []
	rewards = []
	records = [0]
	
	for ep in range(EPISODES):
		for i in count():

			state = game.get_state()
			action = agent.choose_action(state)
			score, reward, done = game.play_step(action)

			scores.append(score)
			rewards.append(reward)
			if reward > records[-1]:
				records.append(reward)
			else:
				records.append(records[-1])
			if done:
				break
			input()

if __name__ == '__main__':
	main()
