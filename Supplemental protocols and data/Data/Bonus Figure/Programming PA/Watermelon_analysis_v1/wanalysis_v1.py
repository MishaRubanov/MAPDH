# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 10:03:25 2022

@author: rmish
"""
# import image_processor_v5 as IP
import os
import pandas as pd
import cv2
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics
from PIL import Image
from scipy import ndimage
from skimage.feature import blob_dog
import re
import copy
from matplotlib.colors import LinearSegmentedColormap

image_folder = 'D:\\2.8.22_Josh_fig3patterning\\Set2diff'
label = 'set2'

def automatic_brightness_and_contrast(image, clip_hist_percent=3):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[65536],[0,65536])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def colormapgenerator(list_colors,label = 'mylist', bins = 100):
    cmaps = LinearSegmentedColormap.from_list(
        label,
        list_colors,
        N = 100)
    return cmaps

cy3 = plt.imread('Cy3_4x4.tif')
cy5 = plt.imread('Cy5_4x4.tif')
fam = plt.imread('FAM_4x4.tif')

# new_inferno = cm.get_cmap('Greens', 5)# visualize with the new_inferno colormaps
# # z = new_inferno.colors

gr = cm.get_cmap('Greens',5)
y = cm.Greens_r
ab = 0
blue = 0.2
maxalpha = 1
g1 = (11,142,54)
lgreen = tuple(ti/255 for ti in t)
lgreen = 

#%%to make the list colors better, you can add a 

list_colors = [(0, 0, 0,ab),
               # (0, 0, 0,ab),
               # (0, 0.1, 0.1,0.1),
               (0, .5, blue,maxalpha)] 

darkgreen = colormapgenerator(list_colors)

list_colors = [(0, 0, 0,ab),
               # (0, 0, 0,ab),
               # (0, 0.1, 0,0.1),
               (0, 1, blue,maxalpha)] 
lightgreen = colormapgenerator(list_colors)


list_colors = [(0, 0, 0,ab),
               # (0, 0, 0,ab),
               # (0.1, 0, 0.1,0.1),
               (1, 0, 0.1,maxalpha-.3)] 
red = colormapgenerator(list_colors)

filler, alpha, beta =automatic_brightness_and_contrast(cy3)
cy3norm = cv2.convertScaleAbs(cy3, alpha=alpha, beta=beta) 

filler, alpha, beta =automatic_brightness_and_contrast(cy5,0.0001)
cy5norm = cv2.convertScaleAbs(cy5, alpha=alpha, beta=beta) 

filler, alpha, beta =automatic_brightness_and_contrast(fam)
famnorm = cv2.convertScaleAbs(fam, alpha=alpha, beta=beta) 

r = red(cy3norm,0.5)
lg = lightgreen(famnorm)
dg = darkgreen(cy5norm)
plt.style.use('dark_background')

plt.figure(frameon=False)
z=plt.imshow(cy3norm, cmap = red)
plt.savefig('redonly.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig('redonly.png',format='png',dpi=150,pad_inches=0)
plt.figure(frameon=False)
plt.imshow(famnorm, cmap = lightgreen)
plt.savefig('lgreenonly.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig('lgreenonly.png',format='png',dpi=150,pad_inches=0)
plt.figure(frameon=False)
plt.imshow(cy5norm, cmap = darkgreen)
plt.savefig('dgreenonly.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig('dgreenonly.png',format='png',dpi=150,pad_inches=0)

plt.figure(frameon=False)

extent = 0, 512, 0, 512
im2 = plt.imshow(cy5norm, cmap=darkgreen, extent=extent)
im3 = plt.imshow(famnorm, cmap=lightgreen, extent=extent)
im1 = plt.imshow(cy3norm, cmap=red,extent=extent)

plt.show()
plt.savefig('alltogether.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig('alltogether.png',format='png',dpi=150,pad_inches=0)

#%%masks:
pns = np.load('pixelmasks.npy')
# pns[:,:,3] = pns[:,:,3]/2
rgbs = (27)
list_colors = [(0, 0, 0,ab),
               # (0, 0, 0,ab),
               # (0.1, 0, 0.1,0.1),
               (.98,.5,.45,1)] 
pink = colormapgenerator(list_colors)

cmaps = [lightgreen, red,darkgreen, pink]
for i in range(len(pns[0,0,:])):
    plt.figure()
    plt.imshow(pns[:,:,i],cmap=cmaps[i])
    ax = plt.gca();
    n = 15
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, n+1, 1))
    ax.set_yticklabels(np.arange(1, n+1, 1))
    
    # Minor ticks
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.savefig('pm'+str(i)+'.svg',format='svg',dpi=150,pad_inches=0)
    plt.savefig('pm'+str(i)+'.png',format='png',dpi=150,pad_inches=0)



# plt.figure()
# plt.imshow(pns[:,:,2],cmap=darkgreen)
# plt.savefig('pm3.svg',format='svg',dpi=150,pad_inches=0)
# plt.savefig('pm3.png',format='png',dpi=150,pad_inches=0)

# plt.figure()
# plt.imshow(pns[:,:,3],cmap='pink')
# plt.savefig('pm4.svg',format='svg',dpi=150,pad_inches=0)
# plt.savefig('pm4.png',format='png',dpi=150,pad_inches=0)