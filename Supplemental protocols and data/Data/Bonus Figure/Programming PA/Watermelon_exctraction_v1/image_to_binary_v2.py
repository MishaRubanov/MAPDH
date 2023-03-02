# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:49:16 2022

@author: rmish
check out: https://github.com/bycloudai/pyxelate-video
"""


import matplotlib.pyplot as plt
import scipy.signal
import scipy.misc
import numpy as np
# import cv2.cv as cv
import cv2
from sklearn.cluster import KMeans

#%%Functions:
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


#%%Pixelation:
img = plt.imread('water-melon.png')
pixelart = np.zeros((15,15,3))
for j in range(3):
    cchannel = img[:,:,j]
    binarization = cchannel[9:343,9:343]
    pixelart[:,:,j] = rebin(binarization[2:332,2:332], (15,15))
    # axs[j].imshow(pixelart[:,:,j])
    
pix2 = pixelart.reshape((pixelart.shape[0] * pixelart.shape[1], 3))

#%%color clustering:
clt = KMeans(n_clusters = 6)
clt.fit(pix2)
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
colors = clt.cluster_centers_[1:5]
# np.save('colors.npy',colors)

# colors = np.load('colors.npy')
bar = plot_colors(hist,colors)
plt.imshow(bar)
plt.show()

# # lpink = cv.Scalar

# tplot = [0, 2, 4, 5]
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
#%%mask making::
pixelmasks = np.zeros((15,15,4))

for c in range(4):
    for x in range(15):
        for y in range(15):
            pixtrue = [False,False, False]
            for z in range(3):
                if pixelart[x,y,z] > colors[c][z]*0.75 and pixelart[x,y,z]<colors[c][z]*1.3:
                    pixtrue[z] = True
            if pixtrue[0] == True and pixtrue[1] == True and pixtrue[2] == True:
                pixelmasks[x,y,c] = 1
    axs[c].imshow(pixelmasks[:,:,c])

np.save('pixelmasks.npy', pixelmasks)

# for i in range(1):
    
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lbound = tuple(np.array(np.round(colors[i]*0.9),dtype=int))
    # ubound = tuple(np.array(np.round(colors[i]*1.1),dtype=int))
    # # ubound = 
    # fthresh = cv2.inRange(img,(36, 0, 0), (70, 255,255))
    # axs[c].imshow(fthresh)
    # c+=1


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

