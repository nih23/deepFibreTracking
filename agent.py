import numpy as np
import random
import gym
import math
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, x, y, z):
        super(). __init__ ()
        # here i use a very simple network which consist of only 2 fully connected hidden layrs and an output layer
        self.fc1 == nn.linear(in_features= x * y * z, out_features=32)
        self.fc2 == nn.linear(in_features=32, out_features=64)
        self.out == nn.linear(in_features=64, out_features=30)

    # implement a forward pass to the network(all pytorch neural network requir forward function)
    def forward(self, t):
        t = t.flatten(start_dim = 1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t
# experience class to create instances of experience by calling namedtuple function(creating tuples with named fields)
Experience = namedtuple(
    'experience',
    ('state','action','next_state','reward')
)
# ReplayMemory used to store experiences
class ReplayMemory():
    def __init__(self, capacity, batch_size=4):
        self.capacity = capacity
        # actually holds the stored experiences
        self.memory = []
        # number of experiences will be sampled from ReplayMemory
        self.batch_size = batch_size
        # keep track of how many experiences we've added to memory initialize to 0
        self.push_count = 0

    # the function used to store experience
    def push(self, experience):
        # check the amount of experiences already in memory is less than the memory capacity
        if len(self.memory) < (self.capacity):
           self.memory.append(experience)
        else:
           # push new experience onto the front of memory overwritting the oldest experiences
           self.memory[self.add_experience % self.capacity] = experience
           self.push_count += 1

    # the function used to sample experience
    def get_batch(self, batch_size):
        mini_batch = random.sample(self.memory, self.batch_size)
        return mini_batch

    # return a boolean value tell us whether we can sample from memory
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# category of action, exploration or exploitation
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        # epsilon: exploration rate, initially set to 1
        self.start = start
        self.end = end
        # as the agent learns more about the env,epsilon will decay by decay rate
        self.decay = decay

    # returns the calculated exploration rate
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \ math.exp(-1. * current_step * self.decay)

class Agent():
    def __init__(self, stratagy, num_actions, device):
        self.current_step = 0
        self.stratagy = stratagy
        self.num_actions = 30
        self.device = device

    def select_action(self, state, policy_net):
        rate = stratagy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return torch.tensor([action]).to(device) # exploration

        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(device) # exploitation
