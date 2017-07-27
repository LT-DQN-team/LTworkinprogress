#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:13:39 2017

@author: timothee
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:07:47 2017

@author: timothee
"""

import pygame, sys, time, random
from pygame.locals import *
import numpy as np
import torch
import torch.nn as nn






import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

import torchvision.transforms as T

pygame.init()


WINDOWHEIGHT=64
WINDOWWIDTH=64
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
pygame.display.set_caption('Catch XXL')

 # set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)



class Basket():
    
    def __init__(self):
        
        HEIGHT_AGENT = 4
        THICKNESS_AGENT = 2
        WIDTH_AGENT = 10
        self.agent=list([pygame.Rect(WINDOWWIDTH/2,WINDOWHEIGHT-HEIGHT_AGENT,THICKNESS_AGENT,HEIGHT_AGENT),
           pygame.Rect(0,WINDOWHEIGHT-THICKNESS_AGENT,WIDTH_AGENT,THICKNESS_AGENT),
           pygame.Rect(0,WINDOWHEIGHT-HEIGHT_AGENT,THICKNESS_AGENT,HEIGHT_AGENT)])
        self.agent[1].left=self.agent[0].right
        self.agent[2].left=self.agent[1].right
        
    def moveLeft(self):
        if self.agent[0].left > 0:
            for compound in self.agent:
                compound.move_ip(-4, 0)
            
    def moveRight(self):
        if self.agent[2].right<WINDOWWIDTH:
             for compound in self.agent:
                compound.move_ip(4, 0)
    def drawBasket(self,windowSurface):
        for compound in self.agent:
            pygame.draw.rect(windowSurface,BLACK,compound)
            
    def getPosition(self):
        return self.agent[1].center[0]

class fallingObject(object):
    
    def __init__(self,given_color=BLUE, given_radius = 2):
        self.color=given_color
        self.speed= 2
        self.verti_pos= 0
        self.hori_pos=int(random.random() * (WINDOWWIDTH - 20)) + 10
        self.radius=given_radius
       
        
    def refresh(self,*_):        
        self.verti_pos+=self.speed     
     
    def drawObject(self,windowSurface):       
        pygame.draw.circle(windowSurface,self.color,(self.hori_pos,self.verti_pos),self.radius)
        
    def getPosition(self):
        return (self.hori_pos,self.verti_pos)
    
    def getRadius(self):
        return self.radius
class niceObject(fallingObject):
    
    def __init__(self):
        super(niceObject,self).__init__(GREEN)
        
    def caught(self):
        
        return 2
    
class badObject(fallingObject):
    
    def __init__(self):
        super(badObject,self).__init__(RED)
        
    def caught(self):
        
        return -1
class trackingObject(fallingObject):
    
    def __init__(self):
        super(trackingObject,self).__init__(RED, 2)
        
        self.width = 1
        
    def getRadius(self):
        return self.radius
        
    def refresh(self,basket):
        self.verti_pos+=self.speed 
        self.hori_pos+=self.speed * ((basket.getPosition() > self.hori_pos) * 2 - 1)
                    
    def drawObject(self,windowSurface):       
        pygame.draw.circle(windowSurface,self.color,(self.hori_pos,self.verti_pos),self.radius,self.width)
        
    def caught(self):
        return -1

#class bonusObject:
    
        
class Environment():
    
    def __init__(self):
        
        #pygame.time.set_timer(USEREVENT, 1000)   
        self.ballList=[niceObject()]
        
    def refresh(self, basket, windowSurface):
       non_collided=[] 
       reward = 0
       flag=''
       global n_frames
       
       if n_frames%20==0:
           
           if random.random() < 0.3:
               self.ballList.append(badObject())
           else:
#               if random.random() <= 1:
                   self.ballList.append(niceObject()) 
#                   else:
#                       self.ballList.append(trackingObject())
                   
       for F_object in self.ballList:
           
           F_object.drawObject(windowSurface)
           F_object.refresh(basket)
           
           
           if self.checkCollision(F_object,basket):
               reward+= F_object.caught()
               flag='Caught'
               
               
           elif ((F_object.getPosition()[1]) > WINDOWHEIGHT)&(isinstance(F_object,niceObject)):
               flag='Missed'
               reward -=1
               
           else:
               non_collided.append(F_object)
       self.ballList=non_collided      
       
       return reward,flag
          
           
    def checkCollision(self,F_object,basket):
         
        
         if (((F_object.getPosition()[1]) >= basket.agent[0].top) &
            ((F_object.getPosition()[0]) <= basket.agent[2].left) &
            ((F_object.getPosition()[0]) >= basket.agent[0].right)):
               return True
         else:
                
                
               return False
               
def getEnv():
        array = np.array(pygame.surfarray.pixels3d(windowSurface))
        return array
       


basket = Basket()
    
env = Environment()
clock = pygame.time.Clock()
n_frames=0

def main(action):     
    reward=0
    
    
    
    global n_frames
    n_frames+=1   
          
    windowSurface.fill(WHITE)   
    
       
    
    
    if (action==0):
            basket.moveLeft()
            
    
    elif (action==1):
            basket.moveRight()
            
    reward=env.refresh(basket,windowSurface)        
    basket.drawBasket(windowSurface)
    pygame.display.update()
    clock.tick(50)
    
    
    return reward
    
    