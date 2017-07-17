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
pygame.display.set_caption('Hello world!')

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
                compound.move_ip(-1,0)
            
    def moveRight(self):
        if self.agent[2].right<WINDOWWIDTH:
             for compound in self.agent:
                compound.move_ip(1,0)
    def drawBasket(self,windowSurface):
        for compound in self.agent:
            pygame.draw.rect(windowSurface,RED,compound)
            
class FallingObject():
    
    def __init__(self,windowSurface,given_color=BLUE):
        self.color=given_color
        self.verti_pos=0
        self.hori_pos=int(random.random()*WINDOWWIDTH)
        print(int(random.random()*WINDOWWIDTH))
        
    def refresh(self):        
            self.verti_pos+=1      
     
    def drawObject(self,windowSurface):       
            pygame.draw.circle(windowSurface,self.color,(self.hori_pos,self.verti_pos),5)
    
    
        
#define agent
basket=Basket()
test=FallingObject(windowSurface,BLUE)




 # run the game loop
while True:
    # check for the QUIT 
    
    test.refresh()
    if (pygame.key.get_pressed()[pygame.K_LEFT])!=0:
        basket.moveLeft()
    
    if (pygame.key.get_pressed()[pygame.K_RIGHT])!=0:
       basket.moveRight()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
            
    
    windowSurface.fill(WHITE)
    test.drawObject(windowSurface)
    basket.drawBasket(windowSurface)
    pygame.display.update()
    