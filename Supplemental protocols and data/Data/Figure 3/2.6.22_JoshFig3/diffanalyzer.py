# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 20:05:41 2022

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
import re

image_folder = 'D:\\2.8.22_Josh_fig3patterning\\Set2diff'
label = 'set2'

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
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

#%% Sorting:
images = [img for img in os.listdir(image_folder)]
df = pd.DataFrame(index=range(3),columns=range(29))
times = np.zeros(29)
c=0
for i in range(len(images)):
    im = images[i]
    y,t,d= re.findall('[0-9]+', im)[-3:]
    if 'Cy3' in im:
        df.iloc[0,int(y)] = images[i]
        times[c] = int(t)
        c+=1
    if 'Cy5' in im:
        df.iloc[1,int(y)] = images[i]
    if 'GFP-FAM' in im:
        df.iloc[2,int(y)] = images[i]

times = np.array(np.round(np.sort(times)/60),dtype='int')

#%%Movie and image plotting:
# adj = -1
# hpercent=0.001
# frame = cv2.imread(os.path.join(image_folder, df.iloc[0,0]),-1)
# height, width = frame.shape

# cy3init = cv2.imread(os.path.join(image_folder, df.iloc[0,0]),-1)
# filler,alphacy3, betacy3 = automatic_brightness_and_contrast(cy3init,clip_hist_percent=hpercent)    
# cy5init = cv2.imread(os.path.join(image_folder, df.iloc[1,0]),-1)
# filler,alphacy5, betacy5 = automatic_brightness_and_contrast(cy5init,clip_hist_percent=hpercent)    
# attoinit = cv2.imread(os.path.join(image_folder, df.iloc[2,0]),-1)
# filler,alphaatto, betaatto = automatic_brightness_and_contrast(attoinit,clip_hist_percent=hpercent)    

# video = cv2.VideoWriter(label+'try2.mp4', 0, 3, frameSize=(width,height),fps=1)

# font = cv2.FONT_HERSHEY_SIMPLEX          # font  
# fontScale = 2
# color = (255,255,255)
# thickness = 4   # Line thickness of 2 px     
# td = 1

# for i in range(12):
#     imCy3 = cv2.imread(os.path.join(image_folder, df.iloc[0,i]),-1)
#     imCy3max = cv2.convertScaleAbs(imCy3, alpha=alphacy3, beta=betacy3)   

#     imCy5 = cv2.imread(os.path.join(image_folder, df.iloc[1,i]),-1)
#     imCy5max = cv2.convertScaleAbs(imCy5, alpha=alphacy5, beta=betacy5)   

#     imAtto = cv2.imread(os.path.join(image_folder, df.iloc[2,i]),-1)
#     imAttomax = cv2.convertScaleAbs(imAtto, alpha=alphaatto, beta=betaatto)   


#     imtot = np.zeros([1024,1024,3],dtype='uint8')
#     imtot[:,:,0]= imCy5max#(imCy3/256).astype('uint8')
#     imtot[:,:,1]= imCy3max#imCy5max#(imCy5/256).astype('uint8')
#     imtot[:,:,2]= imAttomax#(imAtto/256).astype('uint8')
#     imtime = cv2.putText(imtot,str(times[i])+' mins',(10,int(height/10)),font, fontScale, color, thickness, cv2.LINE_AA, False)
#     imrect = cv2.rectangle(imtime,(int(800/td),int(925/td)),(int(963/td),int(950/td)),(255,255,255),-1)
#     video.write(imrect)   
#     plt.imsave(label+'_cycle_'+str(i)+'.png',imrect)

# video.release()
# cv2.destroyAllWindows()


#%% Mask making:

imCy3 = cv2.imread(os.path.join(image_folder, df.iloc[0,0]),-1)
Cy3mask = imCy3> 20000
imCy5 = cv2.imread(os.path.join(image_folder, df.iloc[1,0]),-1)
Cy5mask = imCy5 > 20000
imAtto = cv2.imread(os.path.join(image_folder, df.iloc[2,0]),-1)
Attomask = imAtto > 8000

rCy3 = Cy3mask & ~Attomask & ~Cy5mask
rCy5 = ~Cy3mask & ~Attomask & Cy5mask
rAtto = ~Cy3mask & Attomask & ~Cy5mask
rtot = Cy3mask & Attomask & Cy5mask

plt.imsave(label+'_Cy3mask'+'.png',rCy3)
plt.imsave(label+'_Cy5mask'+'.png',rCy5)
plt.imsave(label+'_Attomask'+'.png',rAtto)
plt.imsave(label+'_rtot'+'.png',rtot)


#%% mean plotting for non-multi-colored gel:
nps = 6
cy3mean = np.zeros(nps)
cy5mean = np.zeros(nps)
attomean = np.zeros(nps)

for i in range(nps):
    imCy3 = cv2.imread(os.path.join(image_folder, df.iloc[0,i]),-1)
    imCy5 = cv2.imread(os.path.join(image_folder, df.iloc[1,i]),-1)
    imAtto = cv2.imread(os.path.join(image_folder, df.iloc[2,i]),-1)
    
    cy3mean[i] = np.mean(imCy3[rCy3])
    cy5mean[i] = np.mean(imCy5[rCy5])
    attomean[i] = np.mean(imAtto[rAtto])
    
    


plt.figure(dpi=150)
plt.plot(times[:nps],cy3mean/max(cy3mean),linewidth=4,c='g')
plt.plot(times[:nps],cy5mean/max(cy5mean),linewidth=4,c='b')
plt.plot(times[:nps],attomean/max(attomean),linewidth=4,c='r')
fs = 13
plt.ylabel('Normalized Fluorescence',fontsize=fs,weight='bold')
plt.xlabel('time (mins)',fontsize=fs,weight='bold')
plt.xticks(fontsize=fs,weight='bold')
plt.yticks(fontsize=fs,weight='bold')
ax1 = plt.gca()
# ax1.set_ylim([0, max(meansnear[j,:len(tplot)])+1000])
ax1.set_ylim([-0.1,1.1])
ax1.fontsize = fs
# ax1.legend(['Close to Sender','Far from Sender'],loc='lower right')
plt.tight_layout()
plt.savefig(label+'_means.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig(label+'_means.png',format='png',dpi=150,pad_inches=0)

#%% mean plotting for multi-colored gel:
    
nps = 6
cy3mean = np.zeros(nps)
cy5mean = np.zeros(nps)
attomean = np.zeros(nps)

for i in range(nps):
    imCy3 = cv2.imread(os.path.join(image_folder, df.iloc[0,i]),-1)
    imCy5 = cv2.imread(os.path.join(image_folder, df.iloc[1,i]),-1)
    imAtto = cv2.imread(os.path.join(image_folder, df.iloc[2,i]),-1)
    
    cy3mean[i] = np.mean(imCy3[rtot])
    cy5mean[i] = np.mean(imCy5[rtot])
    attomean[i] = np.mean(imAtto[rtot])
    
    


plt.figure(dpi=150)

plt.plot(times[:nps],cy3mean/max(cy3mean),linewidth=4,c='g')
plt.plot(times[:nps],cy5mean/max(cy5mean),linewidth=4,c='b')
plt.plot(times[:nps],attomean/max(attomean),linewidth=4,c='r')
# plt.plot(times[:nps],cy3mean,linewidth=4,c='g')
# plt.plot(times[:nps],cy5mean,linewidth=4,c='b')
# plt.plot(times[:nps],attomean,linewidth=4,c='r')
fs = 13
plt.ylabel('Normalized Fluorescence',fontsize=fs,weight='bold')
plt.xlabel('time (mins)',fontsize=fs,weight='bold')
plt.xticks(fontsize=fs,weight='bold')
plt.yticks(fontsize=fs,weight='bold')
ax1 = plt.gca()
# ax1.set_ylim([0, max(meansnear[j,:len(tplot)])+1000])
ax1.set_ylim([-0.1,1.1])
ax1.fontsize = fs
# ax1.legend(['Close to Sender','Far from Sender'],loc='lower right')
plt.tight_layout()
plt.savefig(label+'_totmeans.svg',format='svg',dpi=150,pad_inches=0)
plt.savefig(label+'_totmeans.png',format='png',dpi=150,pad_inches=0)