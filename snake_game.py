import pygame
import random
from collections import namedtuple
from time import sleep

pygame.init()
font = pygame.font.SysFont('arial', 25)

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3
	
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 30
SPEED = 20

class SnakeGame:
	
	DISPLAY = True

	def __init__(self, w=48, h=30):

		self.w = w * BLOCK_SIZE
		self.h = h * BLOCK_SIZE
		self.w_blocks = w
		self.h_blocks = h

		# init display
		if SnakeGame.DISPLAY:
			self.display = pygame.display.set_mode((self.w, self.h))
			pygame.display.set_caption('Snake')
			self.clock = pygame.time.Clock()

	def reset(self):
		# init game state
		self.direction = random.randint(0, 3)
		
		self.head = Point(random.randint(1, self.w_blocks - 2), random.randint(1, self.h_blocks - 2))
		self.snake = [self.head,
					  Point(self.head.x - 1, self.head.y),
					  Point(self.head.x - 2, self.head.y)]
		
		self.score = 0
		self.food = None
		self._place_food()
		self.frame = 0
		
	def _place_food(self):
		x = random.randint(0, self.w_blocks)
		y = random.randint(0, self.h_blocks)
		self.food = Point(x, y)
		if self.food in self.snake:
			self.score += 1
			self._place_food()
		
	def play_step(self, action):
		# 1. collect user input
		if action[1]:
			self.direction = (self.direction + 1) % 4
		elif action[2]:
			self.direction = (self.direction - 1) % 4
		
		# 2. move
		self._move(self.direction) # update the head
		self.snake.insert(0, self.head)
		
		# 3. check if game over
		game_over = False
		if self._is_collision():
			game_over = True
			return game_over, self.score
			
		# 4. place new food or just move
		if self.head == self.food:
			self.score += 1
			self._place_food()
		else:
			self.snake.pop()
		
		# 5. update ui and clock
		self._update_ui()
		self.clock.tick(SPEED)
		# 6. return game over and score
		return game_over, self.score
	
	def _is_collision(self):
		# hits boundary or itself
		return self.head.x >= self.w_blocks or self.head.x < 0 or self.head.y >= self.h_blocks or self.head.y < 0\
			or self.head in self.snake[1:]
		
	def _update_ui(self):
		self.display.fill(BLACK)
		
		for pt in self.snake:
			pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
			pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))
			
		pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE - 2, BLOCK_SIZE - 2))
		
		text = font.render("Score: " + str(self.score), True, WHITE)
		self.display.blit(text, [0, 0])
		pygame.display.flip()
		
	def _move(self, direction):
		x = self.head.x
		y = self.head.y
		if direction == RIGHT:
			x += 1
		elif direction == LEFT:
			x -= 1
		elif direction == DOWN:
			y += 1
		elif direction == UP:
			y -= 1
			
		self.head = Point(x, y)
			

if __name__ == '__main__':
	game = SnakeGame()
	
	# game loop
	# sleep(2)
	while True:
		game_over, score = game.play_step()
		
		if game_over == True:
			break
		
	print('Final Score', score)
		
		
	pygame.quit()