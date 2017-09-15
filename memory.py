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
        data = list(args)
        
        for d in data:
           
            d = (d * 255).type(torch.ByteTensor).cpu() if(d is not None) else None
           
        self.memory[self.position] = Transition(*data)
        self.position = (self.position + 1)% self.capacity
    
    def sample(self, batch_size):
        samples=(random.sample(self.memory , batch_size))
        for s in samples:
            for t in s:
                t = t/255
        
        return samples
    
    def __len__(self):
        return len(self.memory)
    
    
