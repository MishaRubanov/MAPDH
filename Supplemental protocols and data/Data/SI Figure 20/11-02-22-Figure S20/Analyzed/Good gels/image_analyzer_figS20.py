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
afact = 1
bfact = 1

rc='ed2024'
redc = hextorgb(rc)
red = tuple(ti/255 for ti in redc)

gc='2ca02cf'
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

g1 = skimage.io.imread('atto1.tif')
b1 = skimage.io.imread('cy3_1.tif')
r1 = skimage.io.imread('tye_1.tif')
g2 = skimage.io.imread('atto2.tif')
b2 = skimage.io.imread('cy3_2.tif')
r2 = skimage.io.imread('tye_2.tif')
g3 = skimage.io.imread('atto3.tif')
b3 = skimage.io.imread('cy3_3.tif')
r3 = skimage.io.imread('tye_3.tif')

#%%
plt.figure()
plt.style.use('dark_background')

filler, alpha, beta =automatic_brightness_and_contrast(b3)
bnorm = cv2.convertScaleAbs(b3, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(bnorm, cmap = blue)

filler, alpha, beta =automatic_brightness_and_contrast(r3)
rnorm = cv2.convertScaleAbs(r3, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(rnorm, cmap = red)

filler, alpha, beta =automatic_brightness_and_contrast(g3)
gnorm = cv2.convertScaleAbs(g3, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(gnorm, cmap = green)
plt.axis('off')
plt.savefig('gel3.svg')

#%%
plt.figure()
plt.style.use('dark_background')

filler, alpha, beta =automatic_brightness_and_contrast(b2)
bnorm = cv2.convertScaleAbs(b2, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(bnorm, cmap = blue)

filler, alpha, beta =automatic_brightness_and_contrast(r2)
rnorm = cv2.convertScaleAbs(r2, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(rnorm, cmap = red)

filler, alpha, beta =automatic_brightness_and_contrast(g2)
gnorm = cv2.convertScaleAbs(g2, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(gnorm, cmap = green)
plt.axis('off')
plt.savefig('gel2.svg')

#%%
plt.figure()
plt.style.use('dark_background')

filler, alpha, beta =automatic_brightness_and_contrast(b1)
bnorm = cv2.convertScaleAbs(b1, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(bnorm, cmap = blue)

filler, alpha, beta =automatic_brightness_and_contrast(r1)
rnorm = cv2.convertScaleAbs(r1, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(rnorm, cmap = red)

filler, alpha, beta =automatic_brightness_and_contrast(g1)
gnorm = cv2.convertScaleAbs(g1, alpha=alpha*afact, beta=beta*bfact) 
plt.imshow(gnorm, cmap = green)
plt.axis('off')
plt.savefig('gel1.svg')

# #%% plot all 3 filtered
# norm3 = np.zeros((512,512,3))
# norm3[:,:,0] = bnorm
# norm3[:,:,1] = rnorm
# norm3[:,:,2] = gnorm
# # norm3 = np.array((bnorm,rnorm,gnorm))
# import skimage.morphology as morph
# from skimage.color import rgb2gray

# plt.figure()
# plt.imshow(bnorm, cmap = blue)
# plt.imshow(rnorm, cmap = red)
# plt.imshow(gnorm, cmap = green)
# plt.savefig('all3_filtered.svg')


# # #%% plot all 3
# # norm3 = np.zeros((512,512,3))
# # norm3[:,:,0] = bnorm
# # norm3[:,:,1] = rnorm
# # norm3[:,:,2] = gnorm
# # # norm3 = np.array((bnorm,rnorm,gnorm))
# # import skimage.morphology as morph
# # from skimage.color import rgb2gray

# # norm = rgb2gray(norm3)
# # z = morph.white_tophat(norm,footprint=morph.disk(25))
# # z1 = z>(np.max(z)*0.5)
# # plt.imshow(z1)
# # z2 = 1-z1

# # plt.figure()
# # plt.imshow(bnorm*z2, cmap = blue)
# # plt.imshow(rnorm*z2, cmap = red)
# # plt.imshow(gnorm*z2, cmap = green)

# # plt.figure()
# # plt.imshow(norm)

# # plt.figure()
# # ao = morph.area_opening(norm,area_threshold=500,connectivity=2)

# # #%%
# # o = morph.closing(norm,footprint=morph.disk(2))
# # plt.imshow(o)


# #%% plot all 2
# plt.figure()
# plt.style.use('dark_background')

# filler, alpha, beta =automatic_brightness_and_contrast(b2)
# bnorm = cv2.convertScaleAbs(b2, alpha=alpha*afact, beta=beta*bfact) 
# plt.imshow(bnorm, cmap = blue)

# filler, alpha, beta =automatic_brightness_and_contrast(r2)
# rnorm = cv2.convertScaleAbs(r2, alpha=alpha*afact, beta=beta*bfact) 
# plt.imshow(rnorm, cmap = red)

# filler, alpha, beta =automatic_brightness_and_contrast(g2)
# gnorm = cv2.convertScaleAbs(g2, alpha=alpha*afact, beta=beta*bfact) 
# plt.imshow(gnorm, cmap = green)
# plt.savefig('all2.svg')

# #%% plot all 2 filtered
# norm3 = np.zeros((512,512,3))
# norm3[:,:,0] = bnorm
# norm3[:,:,1] = rnorm
# norm3[:,:,2] = gnorm
# # norm3 = np.array((bnorm,rnorm,gnorm))
# import skimage.morphology as morph
# from skimage.color import rgb2gray

# norm = rgb2gray(norm3)
# z = morph.white_tophat(norm,footprint=morph.disk(35))
# z1 = z>(np.max(z)*0.9)
# plt.imshow(z1)
# z2 = 1-z1

# plt.figure()
# plt.imshow(bnorm*z2, cmap = blue)
# plt.imshow(rnorm*z2, cmap = red)
# plt.imshow(gnorm*z2, cmap = green)
# plt.savefig('all2_filtered.svg')


# #%% plot all 1
# plt.figure()
# plt.style.use('dark_background')

# filler, alpha, beta =automatic_brightness_and_contrast(b1)
# bnorm = cv2.convertScaleAbs(b1, alpha=alpha*afact, beta=beta*bfact) 
# plt.imshow(bnorm, cmap = blue)

# filler, alpha, beta =automatic_brightness_and_contrast(r1)
# rnorm = cv2.convertScaleAbs(r1, alpha=alpha*afact, beta=beta*bfact) 
# plt.imshow(rnorm, cmap = red)

# filler, alpha, beta =automatic_brightness_and_contrast(g1)
# gnorm = cv2.convertScaleAbs(g1, alpha=alpha*afact, beta=beta*bfact) 
# plt.imshow(gnorm, cmap = green)
# plt.savefig('all1.svg')

# #%% plot all 1 filtered
# norm3 = np.zeros((512,512,3))
# norm3[:,:,0] = bnorm
# norm3[:,:,1] = rnorm
# norm3[:,:,2] = gnorm
# # norm3 = np.array((bnorm,rnorm,gnorm))
# import skimage.morphology as morph
# from skimage.color import rgb2gray

# norm = rgb2gray(norm3)
# z = morph.white_tophat(norm,footprint=morph.disk(25))
# z1 = z>(np.max(z)*0.9)
# plt.imshow(z1)
# z2 = 1-z1

# plt.figure()
# plt.imshow(bnorm*z2, cmap = blue)
# plt.imshow(rnorm*z2, cmap = red)
# plt.imshow(gnorm*z2, cmap = green)
# plt.savefig('all1_filtered.svg')

