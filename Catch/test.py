#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:33:45 2017

@author: timothee
"""

import Catch
import random
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple


#is_ipython='inline' in matplotlib.get_backend()
#
#if is_ipython:
#    from IPython import display

#ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Transition = namedtuple('Transition',('state','action','next_state','reward'))
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory = []
        self.position=0
        
    def push(self,*args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1)% self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory , batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

"""Training"""
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

#if use_cuda:
#    model.cuda()
    
optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. *steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1]
    else:
        return LongTensor([[random.randrange(2)]])
    
episode_durations= []

#def plot_durations():
#    plt.figure(2)
#    plt.clf()
#    durations_t = torch.FloatTensor(episode_durations)
#    plt.title('Training...')
#    plt.xlabel('Episode')
#    plt.ylabel('Duration')
#    plt.plot(durations_t.numpy())
#    
#    if len(durations_t) >= 100:
#        means = durations_t.unfold(0,100,1).mean(1).view(-1)
#        means = torch.cat((torch.zeros(99), means))
#        plt.plot(means.numpy())
#
#    plt.pause(0.001)  # pause a bit so that plots are updated
#    if is_ipython:
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
#    
#last_sync = 0
#

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
num_frames = 1000
for i_frames in range(num_frames):
    # Initialize the environment and state
   
    last_screen = Catch.getEnv()
    current_screen = last_screen
    state = current_screen - last_screen
   
    # Select and perform an action
    action = select_action(state)
    reward = Catch.main(action[0, 0])
    reward = Tensor([reward])

    # Observe new state
    last_screen = current_screen
    current_screen = Catch.getEnv()
    
    next_state = current_screen - last_screen
    

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the target network)
    optimize_model()
        

while True:
    
    Catch.main(input("Command?"))
    print Catch.getEnv()