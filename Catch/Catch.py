#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:07:47 2017

@author: timothee
"""

import pygame, sys, time, random
from pygame.locals import *


pygame.init()


WINDOWHEIGHT=800
WINDOWWIDTH=600
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
        
        HEIGHT_AGENT=30
        THICKNESS_AGENT=10
        WIDTH_AGENT=80
        self.agent=list([pygame.Rect(WINDOWWIDTH/2,WINDOWHEIGHT-HEIGHT_AGENT,THICKNESS_AGENT,HEIGHT_AGENT),
           pygame.Rect(0,WINDOWHEIGHT-THICKNESS_AGENT,WIDTH_AGENT,THICKNESS_AGENT),
           pygame.Rect(0,WINDOWHEIGHT-HEIGHT_AGENT,THICKNESS_AGENT,HEIGHT_AGENT)])
        self.agent[1].left=self.agent[0].right
        self.agent[2].left=self.agent[1].right
        
    def moveLeft(self):
        if self.agent[0].left>0:
            for compound in self.agent:
                compound.move_ip(-10,0)
            
    def moveRight(self):
        if self.agent[2].right<WINDOWWIDTH:
             for compound in self.agent:
                compound.move_ip(10,0)
    def drawBasket(self,windowSurface):
        for compound in self.agent:
            pygame.draw.rect(windowSurface,RED,compound)
            
class fallingObject(object):
    
    def __init__(self,windowSurface,given_color=BLUE,given_speed=5):
        self.color=given_color
        self.speed=given_speed
        self.verti_pos=0
        self.hori_pos=int(random.random()*WINDOWWIDTH)
        self.radius=5
       
        
    def refresh(self):        
            self.verti_pos+=self.speed     
     
    def drawObject(self,windowSurface):       
            pygame.draw.circle(windowSurface,self.color,(self.hori_pos,self.verti_pos),self.radius)
class niceObject(fallingObject):
    
    def __init__(self,windowsSurface,given_speed=5):
        super(niceObject,self).__init__(windowSurface,GREEN,given_speed)
        
    def caught():
        
        return 1
    
class badObject(fallingObject):
    
    def __init__(self,windowsSurface,given_speed=5):
        super(badObject,self).__init__(windowSurface,BLACK,given_speed)
        
    def caught():
        
        return -1
    
class Environment():
    
    def __init__(self):
        
        pygame.time.set_timer(USEREVENT,1000)   
        self.ballList=[niceObject(windowSurface)]
        
    def refresh(self, basket, windowSurface):
       for event in pygame.event.get():
           
           if event.type == QUIT:
            pygame.quit()
            sys.exit()
           if event.type == USEREVENT:
               if random.random()<0.5:
                   self.ballList.append(badObject(windowSurface))
               else:
                   self.ballList.append(niceObject(windowSurface)) 
       for F_object in self.ballList:
           F_object.drawObject(windowSurface)
           F_object.refresh()
           if (F_object.verti_pos - F_object.radius) > WINDOWHEIGHT:
                #print('Deleted')
                del F_object
           elif (((F_object.verti_pos + F_object.radius) == basket.agent[1].top) &
            ((F_object.hori_pos + F_object.radius) <= basket.agent[2].left) &
            ((F_object.hori_pos - F_object.radius) >= basket.agent[0].right)):
               del F_object 
               return True
       if (pygame.key.get_pressed()[pygame.K_LEFT])!=0:
            basket.moveLeft()
    
       if (pygame.key.get_pressed()[pygame.K_RIGHT])!=0:
            basket.moveRight()
       
        
#define agent
basket=Basket()
test=[badObject(windowSurface)]
env = Environment()

clock=pygame.time.Clock()



 # run the game loop
while True:
    # check for the QUIT 
    
 
    
    
   
        
    windowSurface.fill(WHITE)   
    collision=env.refresh(basket,windowSurface)
    if collision:
        print('Collision detected')       
    
    
    
    basket.drawBasket(windowSurface)
    pygame.display.update()
    clock.tick(50)
    