# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:42:13 2020

@author: rmish
"""


import image_processor_v5 as IP
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
image_folder = 'D:\\2.8.22_Josh_fig3patterning\\Set1diff'
numgels =4
hourstoplot=100 

# dark_im = cv2.imread('darkim_cy3_400ms.tif',-1)
# flat_im = cv2.imread('pos_9_cycle_2_time_43.71.tiff',-1)



# image_folder = 'D:\\1.26.22_rO1_v5_gradient_v1\\4_RNAse'
images_FAM = [img for img in os.listdir(image_folder)]# if img.endswith(".tiff")]

# IP.renamer(image_folder,images_FAM,newt=-128508)

im_pd_FAM,t1 = IP.imSorter(images_FAM,numgels,'')

# t1[-1,:] = 10*3600*np.ones(t1.shape[1])
pts = np.load('pts_v1.npy')

video_name = 'sr_cy3'+str(hourstoplot)+'_'
t_toshow = np.round(t1[:,0]/3600,2)
IP.videoMaker(image_folder,video_name, im_pd_FAM,timer=t_toshow,htp=hourstoplot,td=2,\
              framerate = 7,adj=50,hpercent=0.1)#,darkim = dark_im, flatim = flat_im)
profile_name = 'profs_SR_v5_1_prelim_24hrs'
totsmagots = t_toshow[t_toshow<hourstoplot]
# IP.interactive_profile_plotter(im_pd_FAM,image_folder,profile_name,t1,tot=len(totsmagots),divide=1,pts=pts)

gelstom = [0,1,2]#,3,4,5]#,6,7]

z1 =IP.interactive_circle_meaner(im_pd_FAM,gelstom,image_folder,profile_name,t1,sizecirc = 15,\
                                    numcircs=1,ngels = numgels)#,pts=pts)

#%%Plotting means
meansnear = z1#[0]

t2 = t1[:,0]/3600 
tplot = t2[t2<hourstoplot]

gelstoplot = [0,1,2,3,4]
plt.figure(dpi=150)
for j in range(len(gelstom)-1):
    # plt.figure(dpi=150)
    plt.plot(tplot,meansnear[j,:len(tplot)],linewidth=4)#,c='r')
    fs = 13
    plt.ylabel('RFU',fontsize=fs,weight='bold')
    plt.xlabel('time (hr)',fontsize=fs,weight='bold')
    plt.xticks(fontsize=fs,weight='bold')
    plt.yticks(fontsize=fs,weight='bold')
    ax1 = plt.gca()
    ax1.set_ylim([0, max(meansnear[j,:len(tplot)])+1000])
    ax1.set_xlim([-0.1,30])
    ax1.fontsize = fs
    ax1.legend(['Close to Sender','Far from Sender'],loc='lower right')
    # plt.title('Gel '+str(gelstom[j])+'Mean Fluorescence')
    plt.tight_layout()
    plt.savefig('Gelxlim'+str(gelstom[j])+' meanstogLeg.svg',format='svg',dpi=150,pad_inches=0)
    plt.savefig('Gelxlim'+str(gelstom[j])+' meanstogLeg.png',format='png',dpi=150,pad_inches=0)
    # plt.close()

# =============================================================================
# z1 = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[0,0]),-1)
# z2 = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[0,1]),-1)
# 
# newimfolder = 'D:\\8.28.21_fullhinge_RS_diffgel_lowerdeg_v5_2\\Mainmontage'
# i = 0
# im1 = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[i,0]),-1)
# imleft = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[i,5]),-1)
# imright = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[i,1]),-1)
# 
# imtotleft = IP.image_stitcher(im1,imleft,430)
# imtotot = IP.image_stitcher(imright,imtotleft,430)
# plt.imshow(imtotot)
# 
# # for i in im_pd_FAM.index:
# #     im1 = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[i,0]),-1)
# #     imleft = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[i,5]),-1)
# #     imright = cv2.imread(os.path.join(image_folder, im_pd_FAM.iloc[i,1]),-1)
# #     imtotleft = IP.image_stitcher(im1,imleft,430)
# #     imtotot = IP.image_stitcher(imright,imtotleft,430)
# #     Image.fromarray(imtotot).save(os.path.join(newimfolder,'t'+str(i)+'.tiff'))
# # imtot = IP.image_stitcher(z2,z1,430)
# video_name = 'sr_lmontage'+str(hourstoplot)
# t_toshow = np.round(t1[:,0]/3600,2)
# # plt.imshow(imtot)
# z = np.arange(85)
# # imr =  pd.Series(['imright_'+str(z[i])+'.tiff' for i in z])
# # iml = pd.Series(['imleft_'+str(z[i])+'.tiff' for i in z])
# imt = pd.Series(['t'+str(z[i])+'.tiff' for i in z])
# # IP.stitchedVideoMaker(newimfolder,video_name, imt,timer=t_toshow,htp=hourstoplot,td=2,\
# #               framerate = 7,adj=30,hpercent=0.1)#,darkim = dark_im, flatim = flat_im)
# # # t1[-1,:] = 10*3600*np.ones(t1.shape[1])
# # video_name = 'sr_rmontage'+str(hourstoplot)
# 
# # IP.stitchedVideoMaker(newimfolder,video_name, imr,timer=t_toshow,htp=hourstoplot,td=2,\
# #               framerate = 7,adj=30,hpercent=0.1)#,darkim = dark_im, flatim = flat_im)
# # IP.videoMaker(image_folder,video_name, im_pd_FAM,timer=t_toshow,htp=hourstoplot,td=2,\
# #               framerate = 7,adj=30,hpercent=0.1)#,darkim = dark_im, flatim = flat_im)
# profile_name = 'stitched_profs_v1'
# 
# # pts = np.load('stitched_pts_v1.npy')
# IP.stitched_interactive_profile_plotter(imt,newimfolder,profile_name,t1,tot=10,divide=6,)#pts=pts)
# 
# =============================================================================
# pts = np.load('pts_v1.npy')
# z1 =IP.interactive_circle_meaner(im_pd_FAM,gelstom,image_folder,profile_name,t1,sizecirc = 15,\
                                  # numcircs=1,ngels = numgels)#,pts=pts)
# print('plotting...')

# meansnear = z1#[0]

# t2 = t1[:,0]/3600 
# tplot = t2[t2<hourstoplot]

# gelstoplot = [1,3,5,7]
# plt.figure(dpi=150)


# for j in range(len(gelstom)):
#     plt.figure(dpi=150)
#     plt.plot(tplot,meansnear[j,:len(tplot)],linewidth=4,c='r')
#     fs = 13
#     plt.ylabel('Normalized Counts',fontsize=fs,weight='bold')
#     plt.xlabel('time (hr)',fontsize=fs,weight='bold')
#     plt.xticks(fontsize=fs,weight='bold')
#     plt.yticks(fontsize=fs,weight='bold')
#     ax1 = plt.gca()
#     # ax1.set_ylim([0, 1])
#     ax1.fontsize = fs
#     plt.title('Gel '+str(gelstom[j])+'Mean Fluorescence')
#     plt.savefig('Gel'+str(gelstom[j])+' means.svg',format='svg',dpi=150,pad_inches=0)
#     plt.savefig('Gel'+str(gelstom[j])+' means.png',format='png',dpi=150,pad_inches=0)
#     ax1.legend(['Nearest gel','Middle Gel','Furthest Gel'],loc='lower right')
#     plt.close()



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
# ax1.legend(['0.1s','0.3s','0.5s','1s'])
# # plt.title('Gel '+str(gelstom[j])+'Mean Fluorescence')
# plt.savefig('Gel'+str(gelstom[j])+' means.svg',format='svg',dpi=150,pad_inches=0)
# plt.savefig('Gel'+str(gelstom[j])+' means.png',format='png',dpi=150,pad_inches=0)

# plt.close()


