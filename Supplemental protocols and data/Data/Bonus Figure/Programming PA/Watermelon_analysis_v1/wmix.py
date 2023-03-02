# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:51:40 2022

@author: rmish
"""

import matplotlib.pyplot as plt

z = plt.imread('water-melon_v2.png')

thresh = 0.7
zr = z[:,:,0] > thresh-0.4
zg = z[:,:,1] > thresh
zb = z[:,:,2] > thresh-0.4
ztot = zr & zg & zb
z[ztot] = 0
plt.figure(frameon = False)
plt.imshow(z)
plt.savefig('blackw.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig('blackw.png',format='png',dpi=150,pad_inches=0)