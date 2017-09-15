#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:57 2017

@author: timothee
"""
import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition',('state','action','next_state','reward'))
#identical to Catch
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory = []
        self.position=0
        
    def push(self,*args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
                           
        transition_temp = Transition(*args)
        transitions = Transition((transition_temp.state*255).type(torch.ByteTensor),
                                 transition_temp.action.cpu(),
                                 (transition_temp.next_state*255).type(torch.ByteTensor) if transition_temp.next_state is not None else None ,
                                 transition_temp.reward.cpu())
        self.memory[self.position] = transitions
        
        
        self.position = (self.position + 1)% self.capacity
    
    def sample(self, batch_size):
        samples=(random.sample(self.memory , batch_size))
        #difficult to transform the data back here, will do in optimizer            
        return samples
    
    def __len__(self):
        return len(self.memory)
    
    
