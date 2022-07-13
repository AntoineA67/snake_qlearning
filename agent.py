import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from model import Net
from memory import ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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
LEARNING_RATE = .001

class DQN():
    def __init__(self):
        self.evaluate_net = Net(N_ACTIONS)
        self.target_net = Net(N_ACTIONS)
        self.optimizer = torch.optim.RMSprop(self.evaluate_net.parameters(), lr=LEARNING_RATE, alpha=0.95, eps=0.01) ## 10.24 fix alpha and eps
        self.loss_func = torch.nn.MSELoss()
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.learn_step_counter = 0
        self.evaluate_net.cuda()
        self.target_net.cuda()
    
    def select_action(self, s, epsilon):
        if random.random() > epsilon:
            q_eval = self.evaluate_net.forward(s)
            action = q_eval[0].max(0)[1].cpu().data.numpy() ## 10.21 to cpu
        else:
            action = np.asarray(random.randrange(N_ACTIONS))
        return action

    def store_transition(self, s, a, r, s_):
        self.replay_memory.store(s, a, r, s_)

    def learn(self, ):
        if self.learn_step_counter % TARGET_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step_counter += 1

        s_s, a_s, r_s, s__s = self.replay_memory.sample(BATCH_SIZE)

        q_eval = self.evaluate_net(s_s).gather(1, a_s)
        q_next = self.target_net(s__s).detach()
        q_target = r_s + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

def main():
	pass

if __name__ == '__main__':
	main()
