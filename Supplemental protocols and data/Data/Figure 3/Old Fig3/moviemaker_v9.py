# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:42:13 2020

@author: rmish
"""


import image_processor_v2 as IP
import os
import pandas as pd
import cv2
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from skimage.feature import blob_dog
image_folder = 'D:\\6.29.21_ret_dif_LAP\\3_qwash'
numgels = 10
hourstoplot=6
gelstom = [0,1,2,3,4,5,6,7,8]

images_FAM = [img for img in os.listdir(image_folder) if img.endswith(".tiff")]
# newt = 63037
# IP.renamer(image_folder,images_FAM,newt=7)

im_pd_FAM,t1 = IP.imSorter(images_FAM,numgels,'')

# t1[-1,:] = 10*3600*np.ones(t1.shape[1])
pts = np.load('pts_v1.npy')

video_name = 'LAP_q_v1'+str(hourstoplot)+'_'
t_toshow = np.round(t1[:,0]/3600,2)
IP.videoMaker(image_folder,video_name, im_pd_FAM,timer=t_toshow,htp=hourstoplot,td=4,framerate = 3,adj=1)
profile_name = 'profs_q_v1'
# IP.interactive_profile_plotter(im_pd_FAM,image_folder,profile_name,t1,tot=12,divide=1,pts=pts)
pts = np.load('pts_v1.npy')

# z1 =IP.interactive_circle_meaner(im_pd_FAM,gelstom,image_folder,profile_name,t1,sizecirc = 25,numcircs=1,ngels = numgels,pts=pts)
print('plotting...')

# meansnear = z1#[0]

# t2 = t1[:,0]/3600 
# tplot = t2[t2<hourstoplot]

# gelstoplot = [1,3,5,7]
# gelstoplot = [3]
# plt.figure(dpi=150)

# for j in gelstoplot:
#     plt.plot(tplot,meansnear[j,:len(tplot)],'-*',linewidth=3)
#     fs = 13
# plt.ylabel('Normalized Counts',fontsize=fs,weight='bold')
# plt.xlabel('time (hr)',fontsize=fs,weight='bold')
# plt.xticks(fontsize=fs,weight='bold')
# plt.yticks(fontsize=fs,weight='bold')
# ax1 = plt.gca()
# ax1.set_ylim([0, 0.4])
# ax1.fontsize = fs
# # ax1.legend(['0.1s','0.3s','0.5s','1s'])
# # plt.title('Gel '+str(gelstom[j])+'Mean Fluorescence')
# plt.savefig('Gel'+str(gelstom[j])+' means.svg',format='svg',dpi=150,pad_inches=0)
# plt.savefig('Gel'+str(gelstom[j])+' means.png',format='png',dpi=150,pad_inches=0)

# plt.close()



# # for j in range(len(gelstom)):
# #     plt.figure(dpi=150)
# #     plt.plot(tplot,meansnear[j,:len(tplot)],linewidth=4)
# #     fs = 13
# #     plt.ylabel('Normalized Counts',fontsize=fs,weight='bold')
# #     plt.xlabel('time (hr)',fontsize=fs,weight='bold')
# #     plt.xticks(fontsize=fs,weight='bold')
# #     plt.yticks(fontsize=fs,weight='bold')
# #     ax1 = plt.gca()
# #     ax1.set_ylim([0, 1])
# #     ax1.fontsize = fs
# #     plt.title('Gel '+str(gelstom[j])+'Mean Fluorescence')
# #     plt.savefig('Gel'+str(gelstom[j])+' means.svg',format='svg',dpi=150,pad_inches=0)
# #     plt.savefig('Gel'+str(gelstom[j])+' means.png',format='png',dpi=150,pad_inches=0)

# #     plt.close()
