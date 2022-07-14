from typing import NamedTuple
import pygame
import random
import numpy as np
import torch
# from torch import R

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

BLOCK_SIZE = 30
SPEED = 10

class SnakeGame:
	
	DISPLAY = True
	NORMAL_GAME = False

	def __init__(self, font, w=48, h=30):
		self.font = font
		self.w = w * BLOCK_SIZE
		self.h = h * BLOCK_SIZE
		self.w_blocks = w
		self.h_blocks = h
		self.state = None

		if not SnakeGame.DISPLAY:
			SnakeGame.NORMAL_GAME = False
		else:
			self.display = pygame.display.set_mode((self.w, self.h))
			pygame.display.set_caption('Snake')
			self.clock = pygame.time.Clock()
		self.reset()

	def reset(self):
		if SnakeGame.NORMAL_GAME:
			self.direction = RIGHT
			self.head = Point(self.w_blocks // 2, self.h_blocks // 2)
		else:
			self.direction = random.randint(0, 3)
			self.head = Point(random.randint(1, self.w_blocks - 3), random.randint(1, self.h_blocks - 3))

		self.snake = [self.head,
					  Point(self.head.x + (self.direction == LEFT) - (self.direction == RIGHT), self.head.y + (self.direction == UP) - (self.direction == DOWN)),
					  Point(self.head.x + (self.direction == LEFT) * 2 - (self.direction == RIGHT) * 2, self.head.y + (self.direction == UP) * 2 - (self.direction == DOWN) * 2)]
		print(self.direction, self.snake)
		self.board = np.zeros((self.h_blocks, self.w_blocks), dtype=int)

		# self.board = np.zeros((self.h_blocks + 2, self.w_blocks + 2), dtype=int)
		# self.board[0] = np.ones(self.board.shape[1])
		# self.board[-1] = np.ones(self.board.shape[1])
		# self.board[:, 0] = np.ones(self.board.shape[0])
		# self.board[:, -1] = np.ones(self.board.shape[0])

		self.score = 0
		self.food = None
		self._place_food()
		self.frame = 0
		self.game_over = False
		self.reward = 0

		for p in self.snake:
			self.board[p.y, p.x] = 1
		self.board[self.food.y, self.food.x] = 100
		
	def _place_food(self):
		x = random.randint(0, self.w_blocks - 1)
		y = random.randint(0, self.h_blocks - 1)
		self.food = Point(x, y)
		if self.food in self.snake:
			self._place_food()
		else:
			self.board[y, x] = 100
		
	def play_step(self, action=None):
		if SnakeGame.NORMAL_GAME:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					quit()
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_LEFT and self.direction != RIGHT:
						self.direction = LEFT
					elif event.key == pygame.K_RIGHT and self.direction != LEFT:
						self.direction = RIGHT
					elif event.key == pygame.K_UP and self.direction != DOWN:
						self.direction = UP
					elif event.key == pygame.K_DOWN and self.direction != UP:
						self.direction = DOWN
		else:
			if action[1]:
				self.direction = (self.direction + 1) % 4
			elif action[2]:
				self.direction = (self.direction - 1) % 4

		self._move(self.direction)
		self.snake.insert(0, self.head)
		self.board[self.snake[0].y, self.snake[0].x] = 1

		if self._is_collision():
			self.game_over = True
			self.reward = -10
			return self.score, self.reward, self.game_over

		if self.head == self.food:
			self.score += 1
			self.reward = 10
			self._place_food()
		else:
			self.board[self.snake[-1].y, self.snake[-1].x] = 0
			self.snake.pop()

		if SnakeGame.DISPLAY:
			self._update_ui()
			self.clock.tick(SPEED)

		return self.score, self.reward, self.game_over
	
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
	
	def get_state(self, verbose: bool=False):
		# l2
		# food_dist = np.sqrt(np.power(self.head.x - self.food.x, 2) + np.power(self.head.y - self.food.y, 2))
		# l1
		food_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

		if verbose:
			print(food_dist)
			print('\n'.join([' '.join([str(x) for x in self.board[i]]) for i in range(len(self.board))]))
			print('\n\n')

		return (self.score, self.reward, self.game_over, food_dist, self.board)
	
	def get_env(self):
		return torch.from_numpy(self.board)
		
	def _move(self, direction):
		self.head = Point(self.head.x + (direction == RIGHT) - (direction == LEFT),
						  self.head.y + (direction == DOWN) - (direction == UP))
