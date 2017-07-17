#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:07:47 2017

@author: timothee
"""

import pygame, sys, time
from pygame.locals import *

pygame.init()

WINDOWHEIGHT=800
WINDOWWIDTH=600
HEIGHT_AGENT=100
THICKNESS_BOTTOM=10
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
pygame.display.set_caption('Hello world!')

 # set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#define agent
agent=[pygame.Rect(WINDOWWIDTH/2,WINDOWHEIGHT-HEIGHT_AGENT,20,HEIGHT)]


#define directions
RIGHT=1
LEFT=2

 # run the game loop
while True:
    # check for the QUIT 
    
    if (pygame.key.get_pressed()[pygame.K_LEFT])!=0:
        rectTest=rectTest.move(-1,0)
       
        
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
            
    
    windowSurface.fill(WHITE)
    pygame.draw.rect(windowSurface,RED,rectTest)
    pygame.display.update()