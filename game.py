from typing import NamedTuple
import pygame
import random
import numpy as np
import torch

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
	
class Point(NamedTuple):
	x: int = 0
	y: int = 0

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20


class SnakeGame:
	
	DISPLAY = True

	def __init__(self, font, w=48, h=30, display=True):
		if not display:
			SnakeGame.DISPLAY = False
		self.speed = 1000
		self.font = font
		self.w = w * BLOCK_SIZE
		self.h = h * BLOCK_SIZE
		self.w_blocks = w
		self.h_blocks = h

		if SnakeGame.DISPLAY:
			self.display = pygame.display.set_mode((self.w, self.h))
			pygame.display.set_caption('Snake')
			self.clock = pygame.time.Clock()
		self.reset()

	def reset(self):
		self.direction = random.randint(0, 3)
		self.head = Point(random.randint(1, self.w_blocks - 3), random.randint(1, self.h_blocks - 3))
		self.snake = [self.head,
					  Point(self.head.x + (self.direction == LEFT) - (self.direction == RIGHT), self.head.y + (self.direction == UP) - (self.direction == DOWN)),
					  Point(self.head.x + (self.direction == LEFT) * 2 - (self.direction == RIGHT) * 2, self.head.y + (self.direction == UP) * 2 - (self.direction == DOWN) * 2)]
		self.n_steps = 0
		self.board = torch.zeros((self.h_blocks, self.w_blocks, 3))
		self.score = 0
		self.food = None
		self._place_food()

		self.board[self.head.y, self.head.x, 0] = 1
		for p in self.snake[1:]:
			self.board[p.y, p.x, 1] = 1
		self.board[self.food.y, self.food.x, 2] = 1
	
	def _place_food(self):
		x = random.randint(0, self.w_blocks - 1)
		y = random.randint(0, self.h_blocks - 1)
		f = Point(x, y)
		if f not in self.snake:
			self.food = f
			self.board[y, x, 2] = 1
		else:
			self._place_food()
		
	def play_step(self, action):
		if action is None:
			return self.score, 0, False
		food_dist = np.sqrt(np.power(abs(self.head.x - self.food.x), 2) + np.power(abs(self.head.y - self.food.y), 2))
		self.n_steps += 1
		if action == 1:
			self.direction = (self.direction + 1) % 4
		elif action == 2:
			self.direction = (self.direction - 1) % 4

		self.board[self.head.y, self.head.x, 0] = 0
		self._move(self.direction)
		self.snake.insert(0, self.head)

		new_dist = np.sqrt(np.power(abs(self.head.x - self.food.x), 2) + np.power(abs(self.head.y - self.food.y), 2))
		game_over, reward = False, 0
		# int(new_dist < food_dist) - int(new_dist > food_dist)
		# 2 / (abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y) + 1)
		if self._is_collision() or self.n_steps > 200:
			game_over = True
			reward = -10
			return self.score, reward, game_over

		self.board[self.head.y, self.head.x, 0] = 1
		for i, p in enumerate(self.snake):
			self.board[p.y, p.x, 1] = 1 - i / len(self.snake)
			# self.board[p.y, p.x, 1] = np.maximum(0, 1 -  (i + 1e-15))
		# self.board[:, :, 1] = np.maximum(0, self.board[:, :, 1] - 1 / (len(self.snake) + 1e-15))
		if self.food == self.head:
			self.n_steps = 0
			self.score += 1
			reward = 10
			self.board[self.food.y, self.food.x, 2] = 0
			self._place_food()
		else:
			self.board[self.snake[-1].y, self.snake[-1].x, 1] = 0
			# self.board[self.snake[-1].y, self.snake[-1].x, 1] -= 1 / (len(self.snake) + 1e-15)
			self.snake.pop()

		if SnakeGame.DISPLAY:
			self._update_ui()
			self.clock.tick(self.speed)

		return self.score, reward, game_over
	
	def _is_collision(self):
		return self.head.x >= self.w_blocks or self.head.x < 0 or self.head.y >= self.h_blocks or self.head.y < 0\
			or self.head in self.snake[1:]
		
	def _update_ui(self):
		self.display.fill(BLACK)
		for i, pt in enumerate(self.snake):
			d = i / len(self.snake) * 6
			pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE + d, pt.y * BLOCK_SIZE + d, BLOCK_SIZE - d * 2, BLOCK_SIZE - d * 2))
			pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4 + d, pt.y * BLOCK_SIZE + 4 + d, 12 - d * 2, 12 - d * 2))

		pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE - 2, BLOCK_SIZE - 2))
		text = self.font.render("Score: " + str(self.score), True, WHITE)
		self.display.blit(text, [0, 0])
		pygame.display.flip()
	
	def debug(self):
		board = self.board.numpy()
		print('\n'.join([' '.join([str(x) for x in board[i]]) for i in range(len(board))]))
		print('\n\n')
	
	def get_env(self):
		return self.board.numpy()
		return self.board.unsqueeze(0)
	
	def get_state(self):
		angle = np.arctan((self.head.y - self.food.y) / (self.food.x - self.head.x + 1e-15))
		angle = ((angle + np.pi / 2 * self.direction) - np.pi) / np.pi
		if self.direction == UP:
			sensor = [(j, i) for i in range(2, -1, -1) for j in range(-2, 3)]
		elif self.direction == RIGHT:
			sensor = [(i, j) for i in range(2, -1, -1) for j in range(2, -3, -1)]
		elif self.direction == DOWN:
			sensor = [(j, i) for i in range(-2, 1) for j in range(2, -3, -1)]
		elif self.direction == LEFT:
			sensor = [(i, j) for i in range(-2, 1) for j in range(-2, 3)]

		sensor.pop(-3)
		s = [angle, self.food.x > self.head.x, self.food.x < self.head.x, self.food.y > self.head.y, self.food.y < self.head.y, 1 / (abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y))]
		for x, y in sensor:
			dx = self.head.x + x
			dy = self.head.y - y
			p = Point(dx, dy)
			s.append(float(dx >= self.w_blocks or dx < 0 or dy >= self.h_blocks or dy < 0 or p in self.snake))

		return s
		
	def _move(self, direction):
		self.head = Point(self.head.x + (direction == RIGHT) - (direction == LEFT),
						  self.head.y + (direction == DOWN) - (direction == UP))
