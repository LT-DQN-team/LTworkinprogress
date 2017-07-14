#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:16:21 2017

@author: LT
"""
import pygame as pg

#set up Pygame
pg.init()


windowSurface=pg.display.set_mode((1920,1080),0,0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)



windowSurface.fill(WHITE)
pg.display.quit()