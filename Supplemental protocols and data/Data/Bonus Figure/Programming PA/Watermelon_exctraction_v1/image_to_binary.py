# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:49:16 2022

@author: rmish
check out: https://github.com/bycloudai/pyxelate-video
"""


import cv2
import matplotlib.pyplot as plt
import scipy.signal
import scipy.misc
import numpy as np

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

# read the image file
img = cv2.imread('water-melon.png', -1)
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  
# converting to its binary form
bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
r1 = img[:,:,2]
rreal = r1[9:343,9:343]>100 
r2 = rebin(rreal[2:332,2:332], (15,15))
# plt
# plt.imshow(rreal)
# plt.figure()
# plt.imshow(r2<.1)
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(r1)
axs[1].imshow(rreal)
axs[2].imshow(r2<0.5)

colors = np.load('colors.npy')

pink = colors[0]
dgreen = colors[2]
red = colors[4]
lgreen = colors[5]


#%%
# res = scipy.misc.imresize(r1, (15,15), interp="bilinear")
# K = np.ones([3,3])
# U = scipy.signal.convolve2d(r1, K, mode='same', boundary='wrap')
# # im = Image.fromarray(r2)

# # cv2.imshow("Binary", bw_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# def median_binner(a,bin_x,bin_y):
#     m,n = np.shape(a)
#     strided_reshape = np.lib.stride_tricks.as_strided(a,shape=(bin_x,bin_y,m//bin_x,n//bin_y),strides = a.itemsize*np.array([(m / bin_x) * n, (n / bin_y), n, 1]))
#     return np.array([np.median(col) for row in strided_reshape for col in row]).reshape(bin_x,bin_y)

