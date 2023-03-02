# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:57:00 2022

@author: rmish
"""

from skimage.draw import polygon
from skimage.draw import rectangle
import numpy as np
import matplotlib.pyplot as plt
# img = np.zeros((100, 100), dtype=np.uint8)
# x = 50
# Ax = x/2
# Ay = np.round(np.sqrt(3)*x/2)
# r = np.array([1,x, Ax])
# c = np.array([1, 0, Ay])
# rr, cc = polygon(r, c)
# img[rr, cc] = 1
# plt.close('all')
# plt.imshow(img)

def equil_triangle_mask_generator(h,w,base=100):
    mask1 = np.zeros([h,w],dtype='uint8')
    x = base
    Ax = np.round(x/2)
    Ay = np.round(np.sqrt(3)*x/4)
    cx = h/2
    cy = w/2
    r = np.array([cx,cx-Ax, cx+Ax])
    c = np.array([cy+Ay,cy-Ay,cy-Ay])
    rr, cc = polygon(r, c)
    mask1[rr,cc] = 255
    return np.rot90(mask1)

img = equil_triangle_mask_generator(512,512,base=233)
plt.close('all')
plt.imshow(img)

def square_mask_generator(h,w,ex):
    mid = np.array([h/2,w/2])
    start = mid - ex
    ender = mid + ex
    rr,cc = rectangle(tuple(start),end=tuple(ender),shape=[h,w])
    mask2 = np.zeros((h,w),dtype='uint8')
    mask2[rr.astype('int'),cc.astype('int')] = 255
    return mask2

def rectangle_mask_generator(h,w,lx,ly):
    midx = h/2
    midy = w/2
    startx = midx-lx
    starty = midy-ly
    endx = midx + lx
    endy = midy + ly
    rr,cc = rectangle((startx, starty),end=(endx, endy),shape=[h,w])
    mask2 = np.zeros((h,w),dtype='uint8')
    mask2[rr.astype('int'),cc.astype('int')] = 255
    return mask2

img = rectangle_mask_generator(512,512,30,90)
plt.close('all')
plt.imshow(img)

def plus_mask_generator(h,w,wdist = 90, wthick = 30):
    m1 = rectangle_mask_generator(h,w,wthick,wdist)
    m2 = rectangle_mask_generator(h,w,wdist,wthick)
    return m1 + m2

img = plus_mask_generator(512,512)
plt.close('all')
plt.imshow(img)

def grid_pattern_v2(dist,exposure,slimage,numgels=3, ch=4,inte = 1000):
    x=100
    y=100
    hnum = numgels/2
    xspace = np.linspace(x-(2*dist*hnum),x+(2*dist*hnum),2*numgels+1)
    yspace = np.linspace(y-(dist*hnum),y+(dist*hnum),numgels+1)
    xv,yv = np.meshgrid(xspace,yspace,indexing='ij')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xv,yv,c='b')
    ax1.scatter(x,y,c='r')
    plt.show()
    
    for i in range(len(xspace)):
        for j in range(len(yspace)):
            if (xv[i,j],yv[i,j]) != (x,y):
                core.setXYPosition(xv[i,j],yv[i,j])
                time.sleep(0.5)
                patterning(exposure,slimage,channel=ch,intensity=inte)
                
                