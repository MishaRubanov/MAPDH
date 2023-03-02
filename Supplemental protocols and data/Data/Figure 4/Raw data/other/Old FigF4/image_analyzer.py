# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:56:28 2022

@author: rmish
"""

import skimage
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
from skimage import data
from skimage import color
from skimage import img_as_float
import cv2

def hextorgb(hx):
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

def colormapgenerator(list_colors,label = 'mylist', bins = 100):
    cmaps = LinearSegmentedColormap.from_list(
        label,
        list_colors,
        N = 100)
    return cmaps

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

#%%define colors
ab=0

rc='ed2024'
redc = hextorgb(rc)
red = tuple(ti/255 for ti in redc)

gc='93c954'
greenc = hextorgb(gc)
green = tuple(ti/255 for ti in greenc)

bc = '0084FF'
bluec = hextorgb(bc)
blue = tuple(ti/255 for ti in bluec)

list_colors = [(0, 0, 0,ab),
               red] 
red = colormapgenerator(list_colors)
list_colors = [(0, 0, 0,ab),
               green] 
green = colormapgenerator(list_colors)
list_colors = [(0, 0, 0,ab),
               blue] 
blue = colormapgenerator(list_colors)

#%%load images

g1 = skimage.io.imread('step1_green.tif')
g2 = skimage.io.imread('Green_step 2.tif')
r2 = skimage.io.imread('Red_step 2.tif')
g3 = skimage.io.imread('Green_step3.tif')
b3 = skimage.io.imread('Blue_step3.tif')
r3 = skimage.io.imread('Red_step3.tif')



#%% plot all 3
plt.figure()
plt.style.use('dark_background')


filler, alpha, beta =automatic_brightness_and_contrast(b3)
bnorm = cv2.convertScaleAbs(b3, alpha=alpha*1.2, beta=beta) 
plt.imshow(bnorm, cmap = blue)

filler, alpha, beta =automatic_brightness_and_contrast(r3)
rnorm = cv2.convertScaleAbs(r3, alpha=alpha*1.2, beta=beta) 
plt.imshow(rnorm, cmap = red)

filler, alpha, beta =automatic_brightness_and_contrast(g3)
gnorm = cv2.convertScaleAbs(g3, alpha=alpha*1.2, beta=beta) 
plt.imshow(gnorm, cmap = green)
plt.savefig('all3.svg')

#%%plot 2

plt.figure()
filler, alpha, beta =automatic_brightness_and_contrast(r2)
rnorm = cv2.convertScaleAbs(r3, alpha=alpha*1.2, beta=beta) 
plt.imshow(rnorm, cmap = red)#,extent=extent)

filler, alpha, beta =automatic_brightness_and_contrast(g2)
gnorm = cv2.convertScaleAbs(g2, alpha=alpha*1.2, beta=beta) 
plt.imshow(gnorm, cmap = green)#,extent=extent)
plt.savefig('all2.svg')


#%%plot 1
plt.figure()
filler, alpha, beta =automatic_brightness_and_contrast(g1)
gnorm = cv2.convertScaleAbs(g1, alpha=alpha*1.2, beta=beta) 
plt.imshow(gnorm, cmap = green)#,extent=extent)
plt.savefig('all1.svg')


# plt.imshow(skimage.exposure.rescale_intensity(g3,in_range=tuple(percentiles)), cmap = green)


# plt.imshow(skimage.exposure.rescale_intensity(b3,in_range=tuple(percentiles)), cmap = blue)