# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:31:51 2021

@author: rmish
"""
from pycromanager import Bridge
import matplotlib.pyplot as plt
import numpy as np
import time
import patterning_functions_v4 as PF
h = 684
w = 608
radius = 100
bridge = Bridge(convert_camel_case=False)
core = bridge.get_core()
DMD = core.getSLMDevice()
h = core.getSLMHeight(DMD)
w = core.getSLMWidth(DMD)

PF.init()
pos_list = PF.position_list()

# 1 pixel = 0.45 um
# 100 um gel = 100/0.45 = 222 pixels
rx = 100/0.45

# Circle and Objective Parameters
diam = 100
CF = 0.45
diam_conv = diam / CF
draw_circle = PF.circle_mask_generator(h,w,radius=(diam_conv / 2))

# Square and Objective Parameters
square_side = 50
CF = 0.45
square_conv = square_side / CF
draw_square = PF.square_mask_generator(h,w,ex=square_conv)

# Triangle and Objective Parameters
base_side = 100
CF = 0.45
base_conv = base_side / CF
draw_triangle = PF.equil_triangle_mask_generator(h, w, base=base_conv)

# Rectangle and Objective Parameters
lx_side = 100
ly_side = 50
CF = 0.45
lx_conv = lx_side / CF
ly_conv = ly_side / CF
draw_rectangle = PF.rectangle_mask_generator(h, w, lx=lx_conv, ly=ly_conv)

# Plus and Objective Parameters
width_side = 100/2
thick_side = 40/2
CF = 0.45
width_conv = width_side / CF
thick_conv = thick_side / CF
draw_plus = PF.plus_mask_generator(h, w, wdist=width_conv, wthick=thick_conv)


#%%Patterning:
for i in range(0,len(pos_list)):   
    core.setXYPosition(pos_list[i,0],pos_list[i,1])
    time.sleep(3)
        #%%Setting LED intensity/time:
    core.setProperty('Mightex_BLS(USB)','channel',4)
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',1000)
    time.sleep(0.5)
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0)
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0)
    time.sleep(3)
    
core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0)
    
