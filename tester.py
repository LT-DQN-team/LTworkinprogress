#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:14:52 2017

@author: timothee
"""


from Catch import env
from Catch import main as game
from Catch import getEnv


import torch.nn as nn



import random

import torch


from torch.autograd import Variable
from collections import namedtuple





import matplotlib.pyplot as plt
from IPython import display




Scenario = namedtuple ('Scenario',('green','red','classifier'))
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
model = None
Q_values = []
def get_screen():
    screen = getEnv()
    
    screen = screen.unsqueeze(0).type(Tensor)

    return screen   
    
    

def selectAction(state):#Tested,working
   
        out = model(Variable(state, volatile=True).type(FloatTensor))
        current_scenario = out[2]
        action_index = out[current_scenario].data.max(1)[1].view(1, 1) #get the index of the output with the highest predicted Q-value
        action = model.indexInterpreter(action_index[0,0])#Translate this index into a combination of button activation
        
        return action,action_index #Need to return both, action for VizDoom to understand, and action_index for the memory and later training
    
   
    

def Qtest():
    env.scorePlugged = False
    current_screen = get_screen()
    state = current_screen
    global Q_values
    Q_values = []
    
    for i_frame in range(500):
        
        
        
        action = LongTensor([[random.randint(0, 2)]])     
        _,_,_ = game(action[0, 0])
        state=Variable(Tensor(state))
        output = model(state)
        Q=Scenario(output[0].max(1)[0].data[0,0],
                   output[1].max(1)[0].data[0,0],
                   output[2].data[0,0])
        current_screen = get_screen()
        state=current_screen
        
        Q_values.append(Q)
        
        
    
    
#Performance test will test the classifier as well
def PerformanceTest():
    env.resetBuffer()
    env.scorePlugged = True
    current_screen = get_screen()
    state = current_screen
    for i_frame in range(1000):
        
        action = select_action(state)
        _,_,_ = game(action[0,0])
        current_screen = get_screen()
        state = current_screen
        
    
    
    

G_performance = []
R_performance = []
score = []    



    
def plotValues():
    
    global G_performance
    global R_performance
    global score
    global Q_values
    
    g_caught,g_missed,r_caught,r_missed,net_score= env.g_caught,env.g_missed,env.red_caught,env.red_missed,env.net_score
    
    G_performance.append(g_caught/(g_caught+g_missed)*100)
    R_performance.append(r_caught/(r_caught+r_missed)*100)
    score.append(net_score)    
    
    plt.figure(1)
    plt.clf()
    
    plt.subplot(411)    
    plt.xlabel('Episode number')
    plt.ylabel('%')     
    plt.plot(G_performance,label='Percentage of green balls caught')
    plt.plot(R_performance,label='Percentage of red balls caught')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.subplot(412)
    plt.plot(score)
    plt.xlabel('Episode number')
    plt.ylabel('Net score')
    
    
    if len(Q_values) > 1:
        plt.subplot(413)
            
        Q_values=Scenario(*zip(*Q_values))
        plt.plot(Q_values.green,label = 'Green branch')
        plt.plot(Q_values.red, label = 'Red branch')
        
        plt.xlabel('Frame number in test sequence')
        plt.ylabel('Q Values') 
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        plt.subplot(414)
        plt.plot(Q_values.classifier, label = 'Classifier')
        plt.xlabel('Frame number in test sequence')
        plt.ylabel('Classification')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    display.clear_output(wait=True)
    fig=plt.gcf()
    
    fig.set_size_inches(8, 15)
    display.display(fig)

def testNetwork(ImpModel):
    global model
    model = ImpModel
    env.resetBuffer()
    PerformanceTest()
    Qtest()
    plotValues()