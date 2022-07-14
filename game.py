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

BLOCK_SIZE = 20


class SnakeGame:
	
	DISPLAY = True
	NORMAL_GAME = False

	def __init__(self, font, w=48, h=30):
		self.speed = 1000
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
			self.direction = RIGHT
			self.head = Point(self.w_blocks // 2, self.h_blocks // 2)
			# self.direction = random.randint(0, 3)
			# self.head = Point(random.randint(1, self.w_blocks - 3), random.randint(1, self.h_blocks - 3))

		self.snake = [self.head,
					  Point(self.head.x + (self.direction == LEFT) - (self.direction == RIGHT), self.head.y + (self.direction == UP) - (self.direction == DOWN)),
					  Point(self.head.x + (self.direction == LEFT) * 2 - (self.direction == RIGHT) * 2, self.head.y + (self.direction == UP) * 2 - (self.direction == DOWN) * 2)]
		# print(self.direction, self.snake)
		# self.board = np.zeros((self.h_blocks, self.w_blocks), dtype=torch.cuda.)
		self.n_steps = 0
		self.board = torch.zeros((self.h_blocks, self.w_blocks, 3))
		# self.explored = torch.zeros((self.h_blocks, self.w_blocks))
		# self.food_board = torch.zeros((self.h_blocks, self.w_blocks))

		# self.board = np.zeros((self.h_blocks + 2, self.w_blocks + 2), dtype=int)
		# self.board[0] = np.ones(self.board.shape[1])
		# self.board[-1] = np.ones(self.board.shape[1])
		# self.board[:, 0] = np.ones(self.board.shape[0])
		# self.board[:, -1] = np.ones(self.board.shape[0])

		self.score = 0
		self.init_food()
		# self._place_food()
		self.frame = 0
		self.game_over = False
		self.reward = 0

		for p in self.snake[:-1]:
			self.board[p.y, p.x, 1] = 100
			# self.explored[p.y, p.x] = 1
		self.board[self.head.y, self.head.x, 0] = 100

		for f in self.food:
			# self.food_board[f.y, f.x] = 255
			self.board[f.y, f.x, 2] = 100
			
		# for f in self.food:
		# 	self.board[f.y, f.x] = 100
		
	def init_food(self):
		self.food = {}
		for i in range(1):
			x = random.randint(0, self.w_blocks - 1)
			y = random.randint(0, self.h_blocks - 1)
			f = Point(x, y)
			if self.food.get(f, 0) == 0:
				self.food[f] = 1
			else:
				i -= 1
	
	def _place_food(self):
		x = random.randint(0, self.w_blocks - 1)
		y = random.randint(0, self.h_blocks - 1)
		f = Point(x, y)
		if self.food.get(f, 0) == 0 and f not in self.snake:
			self.food[f] = 1
			self.food_board[y, x] = 255
			self.board[y, x] = 255
		else:
			self._place_food()
		
	def play_step(self, action=None):
		self.n_steps += 1
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
		f = list(self.food)[0]
		# print(f)
		self.reward = 10 / (abs(self.head.x - f.x) + abs(self.head.y - f.y) + 1)
		# self.reward = -1 * self.n_steps + 100 / (abs(self.head.x - f.x) + abs(self.head.y - f.y) + 1)
		# self.reward = 10 / (abs(self.head.x - f.x) + abs(self.head.y - f.y) + 1) - 10 * self.n_steps
		# self.reward = 10 / (abs(self.head.x - f.x) + abs(self.head.y - f.y) + 1) + int(self.explored.sum())
		# self.reward = int(self.explored.sum())
		# self.reward = -self.n_steps // 100 + 1 / (abs(self.head.x - f.x) + abs(self.head.y - f.y) + 1)
		if self.n_steps > 200:
			self.game_over = True
		if self._is_collision():
			self.game_over = True
			self.reward = - 100
			return self.score, self.reward, self.game_over

		for p in self.snake:
			self.board[p.y, p.x] = 50
		self.board[self.snake[0].y, self.snake[0].x] = 100
		# if self.explored[self.snake[0].y, self.snake[0].x]:
			pass
			# self.reward = -2
		else:
			self.explored[self.snake[0].y, self.snake[0].x] = 1
		if self.food.get(self.head):
			# self.speed = 5
			self.n_steps = 0
			self.food.pop(self.head)
			self.score += 1
			self.reward = 100
			self._place_food()
		else:
			self.board[self.snake[-1].y, self.snake[-1].x] = 0
			self.food_board[self.snake[-1].y, self.snake[-1].x] = 0
			self.snake.pop()

		if SnakeGame.DISPLAY:
			self._update_ui()
			self.clock.tick(self.speed)

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
		for f in self.food:
			pygame.draw.rect(self.display, RED, pygame.Rect(f.x * BLOCK_SIZE, f.y * BLOCK_SIZE, BLOCK_SIZE - 2, BLOCK_SIZE - 2))
		text = self.font.render("Score: " + str(self.score), True, WHITE)
		self.display.blit(text, [0, 0])
		pygame.display.flip()
	
	def get_state(self, verbose: bool=False):
		# l2
		# food_dist = np.sqrt(np.power(self.head.x - self.food.x, 2) + np.power(self.head.y - self.food.y, 2))
		# l1
		# food_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

		if verbose:
			# print(food_dist)
			board = self.board.numpy()
			# board = self.board.numpy()
			print('\n'.join([' '.join([str(x) for x in board[i]]) for i in range(len(board))]))
			print('\n\n')

		# return (self.score, self.reward, self.game_over, food_dist, self.board)
	
	def get_env(self):
		# self.food_board[self.head.y, self.head.x] = 100
		# return self.food_board.unsqueeze(0).unsqueeze(0)
		# print(self.board.unsqueeze(0).unsqueeze(0))
		return self.board.unsqueeze(0).unsqueeze(0)
		f = list(self.food)[0]

		s = Point(self.head.x + (self.direction == RIGHT) - (self.direction == LEFT), self.head.y - (self.direction == UP) + (self.direction == DOWN))
		l = Point(self.head.x + (self.direction == UP) - (self.direction == DOWN), self.head.y - (self.direction == LEFT) + (self.direction == RIGHT))
		r = Point(self.head.x + (self.direction == DOWN) - (self.direction == UP), self.head.y - (self.direction == RIGHT) + (self.direction == LEFT))
		ds = int(s.x >= self.w_blocks or s.x < 0 or s.y >= self.h_blocks or s.y < 0 or s in self.snake)
		dl = int(l.x >= self.w_blocks or l.x < 0 or l.y >= self.h_blocks or l.y < 0 or l in self.snake)
		dr = int(r.x >= self.w_blocks or r.x < 0 or r.y >= self.h_blocks or r.y < 0 or r in self.snake)

		return (int(self.direction == RIGHT),int(self.direction == DOWN),int(self.direction == LEFT),int(self.direction == UP), ds, dl, dr, int(f.x > self.head.x), int(f.x < self.head.x), int(f.y > self.head.y), int(f.y < self.head.y))

		return (self.direction, self.score, self.head.x, self.head.y, 10 / (abs(self.head.x - f.x) + abs(self.head.y - f.y) + 1), f.x, f.y)
		return self.board.flatten()
		
	def _move(self, direction):
		self.head = Point(self.head.x + (direction == RIGHT) - (direction == LEFT),
						  self.head.y + (direction == DOWN) - (direction == UP))
