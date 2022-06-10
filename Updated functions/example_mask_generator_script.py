# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:44:48 2022

@author: rmish
"""

import matplotlib.pyplot as plt
import patterning_functions_v7 as PF
h = 684
w = 608

#%%Mask Generator Examples:
moved_rect = PF.rectangle_mask_generator(h,w,200,100,cx=0,cy=100)
plt.imshow(moved_rect,cmap='Greys_r')

ushaped = PF.hollow_rr_mask_generator(h,w,200,200,100,100,cxs=50)
plt.figure()
plt.imshow(ushaped,cmap='Greys_r')

abcmask = PF.message_mask_generator(h,w,message="DNA",fontsize=200)
plt.figure()
plt.imshow(abcmask,cmap='Greys_r')



