# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:53:38 2021

@author: rmish
"""
import cv2
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb
import numpy as np
import re
import pandas as pd
import os
import cv2
import pandas as pd
import numpy as np
from skimage.draw import circle_perimeter
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import cmapy
import cmcrameri.cm as cmc


def circle_detector(img, rlow, rhigh, sig=3, low_thresh=10, high_thresh=50,nump=1,minX = 1, minY= 1):
    edges = canny(img,sigma=sig, low_threshold=low_thresh, high_threshold=high_thresh)
    h_radii = np.arange(rlow,rhigh,5)
    h_res = hough_circle(edges,h_radii)
    accums, x, y, r = hough_circle_peaks(h_res, h_radii,total_num_peaks=nump,min_xdistance=minX,min_ydistance=minY)
    return x, y, r
    
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def circle_mask(img, x, y, r):
    mask = np.zeros(img.shape)
    circy, circx = circle(y[0], x[0], r[0], shape=img.shape)
    mask[circy,circx]=1
    return mask == 1


def im_circle_visualizer(img,x,y,r):
    circy, circx = circle_perimeter(y, x, r,shape=img.shape)
    img[circy, circx] = np.max(img)*2
    return img

def circle_mean_calculator(image_folder,im_pd,minR=100, maxR=200,FoL='last',x=[],y=[],r=[]):
    means = np.zeros(np.asarray(im_pd.shape))

    for j in im_pd.columns:
        if FoL == 'last':
            lastim = cv2.imread(os.path.join(image_folder, im_pd.iloc[-1,j]),-1)
            print('lasting')
        else:
            lastim = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,j]),-1)
        # if x == []:
        x,y,r = circle_detector(lastim, minR, maxR)
        print(x,y,r)
        cvis = im_circle_visualizer(lastim,x[0],y[0],r[0])
        cv2.imwrite('mask_'+str(j)+'.tiff',cvis)
        for i in im_pd.index:        
            im2write = cv2.imread(os.path.join(image_folder, im_pd.iloc[i,j]),-1)
            if r.size > 0: 
                cmask = circle_mask(lastim, x, y, r)
                means[i,j] = np.mean(im2write[cmask])
                # print(means[i,j])

    return means

def slice_list(lst, slice_size):
    if not isinstance(slice_size, int) or slice_size <= 0:
        raise ValueError("slice_size must be a positive integer")

    for i in range(0, len(lst), slice_size):
        yield lst[i:i+slice_size]

def slice_and_group(lst, slice_size):
    slices = {}
    for i,l in enumerate(slice_list(lst, slice_size)):
        slices[i] = l

    return slices 

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


def imSorter(images,numgels,fl):
    for i in range(len(images)):
        im = images[i]
        x,y,t,d= re.findall('[0-9]+', im)
        if int(y) < 10:
            if int(x) < 10:
                images[i] = 'pos_0'+str(x)+'_cycle_00'+str(y)
            else:
                images[i] = 'pos_'+str(x)+'_cycle_00'+str(y)
        elif int(y) >= 10 and int(y) < 100:
            if int(x) < 10:
                images[i] = 'pos_0'+str(x)+'_cycle_0'+str(y)
            else:
                images[i] = 'pos_'+str(x)+'_cycle_0'+str(y)
        elif int(y) >= 100:
            if int(x) < 10:
                images[i] = 'pos_0'+str(x)+'_cycle_'+str(y)
            else:
                images[i] = 'pos_'+str(x)+'_cycle_'+str(y)
                
        images[i] = images[i]+'_time_'+str(t)+'.'+str(d)+'.tiff'
    images.sort()
    dict_im = slice_and_group(images,slice_size=int(len(images)/numgels))
    im_pd = pd.DataFrame.from_dict(dict_im)
    times = np.zeros(im_pd.shape)
    for i in im_pd.columns:
        for ii in im_pd.index:
            im = im_pd.iloc[ii,i]
            x,y,t,d= re.findall('[0-9]+', im)
            times[ii,i]= float(t+'.'+d)
            # ynew = y
            if int(y) != 0:
                ynew = y.lstrip('0')
            else:
                ynew = '0'
                
            if int(x) != 0:
                xnew = x.lstrip('0')
            else:
                xnew = '0'
                
            if fl == '':
                im_pd.iloc[ii,i] = 'pos_'+str(xnew)+'_cycle_'+str(ynew)+'_time_'+str(t)+'.'+str(d)+'.tiff' 
            else:
                im_pd.iloc[ii,i] = fl+'_pos_'+str(xnew)+'_cycle_'+str(ynew)+'_time_'+str(t)+'.'+str(d)+'.tiff'   
     
    return im_pd, times

def renamer(imagefolder, images,newcyc=[],newt = []):
    if newcyc !=[]:        
        for i in range(len(images)):
            im = images[i]
            x,y,t,d= re.findall('[0-9]+', im)
            nc = int(y) + newcyc
            newimname = 'pos_'+str(x)+'_cycle_'+str(nc)+'_time_'+str(t)+'.'+str(d)+'.tiff'
            os.rename(os.path.join(imagefolder,im),os.path.join(imagefolder,newimname))
    elif newt !=[]:
        for i in range(len(images)):
            im = images[i]
            x,y,t,d= re.findall('[0-9]+', im)
            nt = int(t) + newt
            newimname = 'pos_'+str(x)+'_cycle_'+str(y)+'_time_'+str(nt)+'.'+str(d)+'.tiff'
            os.rename(os.path.join(imagefolder,im),os.path.join(imagefolder,newimname))


def videoMaker(image_folder,video_name,im_pd,delay=10,htp=[],adj=-1,timer=[],td=1,framerate = 2):
    frame = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,0]),-1)
    height, width = frame.shape
    cm2 = plt.get_cmap('gist_rainbow')
    font = cv2.FONT_HERSHEY_SIMPLEX          # font  
    fontScale = 1
    thickness = 1   # Line thickness of 2 px     
    color = (255,255,255)
    if timer == []:
        t1 = np.around(np.arange(0,(im_pd.last_valid_index()+1)*delay,delay)/60,decimals=2)  
    else:
        t1 = timer

    if htp != []:
        times = t1[t1<htp]
    else:
        times = t1
    
    for j in im_pd.columns:        
        last_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[adj,j]),-1)
        first_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,j]),-1)

        firstimblur = cv2.blur(first_im,(5,5))
        lastimblur = cv2.blur(last_im,(5,5))
        difim = cv2.subtract(lastimblur,firstimblur)
        filler,alpha, beta = automatic_brightness_and_contrast(firstimblur,clip_hist_percent=0.001)    
        print('gel'+str(j))
        print('td: '+str(td))
        video = cv2.VideoWriter(video_name+str(j)+'.mp4', 0, 3, frameSize=(width,height),fps=framerate)
        for i in im_pd.index[:len(times)]:        
            im1 = cv2.imread(os.path.join(image_folder, im_pd.iloc[i,j]),-1)
            # im22 = cv2.applyColorMap(im2write,cv2.COLORMAP_RAINBOW())
            # z = cm2(im2write)
            # im2 = cv2.subtract(im1,first_im)
            im1blur = cv2.blur(im1,(5,5))
            # im2 = cv2.subtract(im1blur,firstimblur)
            im3 = cv2.convertScaleAbs(im1blur, alpha=alpha, beta=beta)             
            im4 = cv2.applyColorMap(im3,cmapy.cmap(cmc.batlow))
            imcrop = im4[255-128:255+128,255-128:255+128]
            # im41 = cmc.cm.batlow(im3)[:,:,:-1]
            # im5 = cv2.putText(imcrop,str(times[i])+' hrs',(10,int(height/10)+127),font, fontScale, color, thickness, cv2.LINE_AA, False)
            imrect = cv2.rectangle(imcrop,(int(800/td),int(925/td)),(int(963/td),int(950/td)),(255,255,255),-1)
            # plt.imshow(imrect,cmap=cmc.batlow)
            # plt.savefig('gel'+str(j)+'cyc'+str(i)+'.svg',format='svg',dpi=150,pad_inches=0)
            # plt.close()
            cv2.imwrite('gel'+str(j)+'hr'+str(times[i])+'.png',imrect)
            video.write(imrect)   
        
        video.release()
    
    cv2.destroyAllWindows()
    

def interactive_profile_plotter(im_pd,image_folder,profile_name,t1,tot=10,divide=6,pts=[]):
    first_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,0]),-1)
    length = len(first_im)-2
    ums = np.arange(0,length)/length*1250
    t1isolate = np.around(t1[range(0,tot*divide,divide)])/3600
    tpandas = pd.DataFrame(data=t1isolate).round(2).astype('str') + ' hrs'
    profx = np.zeros(im_pd.shape+(length,))
    profy = np.zeros(im_pd.shape+(length,))
    tp = np.zeros((im_pd.shape[1],2))
    ys=cmc.batlow(np.linspace(0,1,tot+1))

    for l in im_pd.columns:
        c=0
        first_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,l]),-1)
        firstimblur = cv2.blur(first_im,(5,5))
        if pts == []:
            plt.imshow(cv2.imread(os.path.join(image_folder, im_pd.iloc[-1,l]),-1))
            plt.waitforbuttonpress()
            tp[l,:] = np.asarray(plt.ginput(1,timeout=-1))
            plt.clf()
            plt.close()            
            cx = tp[l,1]
            cy = tp[l,0]
        else:
            cx = pts[l,1]
            cy = pts[l,0]
        print('profile points are: ',cx,cy)
        for i in im_pd.index:   
            cur_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[i,l]),-1)
            # curimblur = cv2.subtract(cur_im,firstimblur)
            profx[i,l,:] = np.convolve(cur_im[int(cx),:],np.ones(3)/3, mode = 'valid')
            profy[i,l,:] = np.convolve(cur_im[:,int(cy)],np.ones(3)/3, mode = 'valid')
        
    if pts == []:
        np.save('pts_v1',tp)     
    for l in im_pd.columns[:-1]:
        plt.figure(dpi=150)
        c = 0
        for k in range(0,tot*divide,divide):
            plt.plot(ums,profx[k,l,:],color=ys[c])
            c+=1
        fs = 13
        plt.title('Gel '+str(l)+'X-Profile')
        plt.ylabel('NFU',fontsize=fs,weight='bold')
        plt.xlabel('Distance (um)',fontsize=fs,weight='bold')
        plt.xticks(fontsize=fs,weight='bold')
        plt.yticks(fontsize=fs,weight='bold')
        ax1 = plt.gca()
        ax1.fontsize = fs
        plt.legend(tpandas[l],loc='lower left')
        plt.savefig('Gels '+str(l)+' x-profile.svg',format='svg',dpi=150)
        plt.savefig('Gels '+str(l)+' x-profile.png',format='png',dpi=150)
        plt.close()
    
    for l in im_pd.columns:
        ys=cm.viridis(np.linspace(0,1,tot))
        c=0
        plt.figure(dpi=150)
        for k in range(0,tot*divide,divide):
            plt.plot(ums,profy[k,l,:],color=ys[c])
            c+=1
        fs = 13
        plt.title('Gel '+str(l)+'Y-Profile')
        plt.ylabel('RFU',fontsize=fs,weight='bold')
        plt.xlabel('Distance (um)',fontsize=fs,weight='bold')
        plt.xticks(fontsize=fs,weight='bold')
        plt.yticks(fontsize=fs,weight='bold')
        ax1 = plt.gca()
        ax1.fontsize = fs
        plt.legend(tpandas[l],loc='lower left')
        plt.savefig('NFU '+str(l)+' y-profile.svg',format='png',dpi=150)  
        plt.savefig('NFU '+str(l)+' y-profile.png',format='png',dpi=150)

        plt.close()
        
        
        
def interactive_circle_meaner(im_pd,gelstom,image_folder,means_name,t1,sizecirc = 15, numcircs = 3,pts=[],ngels = 6):
    
    meansfar = np.zeros((ngels,im_pd.shape[0]))
    meansmid = np.zeros((ngels,im_pd.shape[0]))
    meansnear = np.zeros((ngels,im_pd.shape[0]))
    # maxsigs = np.zeros((ngels))
    maxsigs = np.load('maxes.npy')
    first_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,0]),-1)
    length = len(first_im)-2
    c=0
    if numcircs == 3:        
        for j in gelstom:
            first_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,j]),-1)
            length = len(first_im)
            if pts == []:                
                plt.figure(dpi=200)
                plt.imshow(cv2.imread(os.path.join(image_folder, im_pd.iloc[-1,j]),-1))
                plt.waitforbuttonpress()
                points = np.asarray(plt.ginput(numcircs,timeout=-1))
                print(points)
            else:
                points = pts[c,:]
            cmaskfar = create_circular_mask(length,length,center=points[2,:],radius=sizecirc)
            cmaskmid = create_circular_mask(length,length,center=points[1,:],radius=sizecirc)
            cmasknear = create_circular_mask(length,length,center=points[0,:],radius=sizecirc)    
            for i in im_pd.index:   
                cur_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[i,j]),-1)
                # curimblur = cv2.subtract(cur_im,firstimblur)
                meansfar[c,i]=np.mean(cur_im[cmaskfar])
                meansmid[c,i]=np.mean(cur_im[cmaskmid])
                meansnear[c,i]=np.mean(cur_im[cmasknear])
            c+=1
            plt.imshow(cmaskfar+cmaskmid+cmasknear,cmap='bone')
            plt.savefig('Gel'+str(j)+' In_Mask.svg',format='svg',dpi=150,pad_inches=0)  
            plt.close()
        return(meansfar,meansmid,meansnear)
    elif numcircs == 1:
        for j in gelstom:
            first_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,j]),-1)
            minsig = 1821
            length = len(first_im)
            if pts == []:                
                plt.figure(dpi=200)
                plt.imshow(cv2.imread(os.path.join(image_folder, im_pd.iloc[-1,j]),-1))
                plt.waitforbuttonpress()
                points = np.asarray(plt.ginput(numcircs,timeout=-1))
                cmasknear = create_circular_mask(length,length,center=points[0,:],radius=sizecirc)
                print(points)
            else:
                points = pts[c,:]
                cmasknear = create_circular_mask(length,length,center=points,radius=sizecirc)    
            for i in im_pd.index:   
                cur_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[i,j]),-1)
                # curimblur = cv2.subtract(cur_im,firstimblur)
                initmean = np.mean(cur_im[cmasknear])
                # if i == 0:
                #     maxsigs[c] = np.max(initmean)
                norm_mean = (initmean-minsig)/(maxsigs[c]-minsig)
                # meansnear[c,i]=np.mean(cur_im[cmasknear])
                meansnear[c,i]=norm_mean
            c+=1
            plt.imshow(cmasknear,cmap='bone')
            plt.savefig('Gel'+str(j)+' In_Mask.svg',format='svg',dpi=150,pad_inches=0)  
            plt.close()
            np.save('maxes.npy', maxsigs, allow_pickle=True, fix_imports=True)
        return(meansnear)

def old_videoMaker(image_folder,video_name,im_pd,delay=10,htp=[],adj=-1,timer=[]):
    frame = cv2.imread(os.path.join(image_folder, im_pd.iloc[0,0]),-1)
    height, width = frame.shape
    
    #100 um = 81.92 pixles
    # 0.8192 pixels/um
    font = cv2.FONT_HERSHEY_SIMPLEX          # font  
    color = (255,255,255)
    fontScale = 3/4
    thickness = 3   # Line thickness of 2 px     
    if timer == []:
        t1 = np.around(np.arange(0,(im_pd.last_valid_index()+1)*delay,delay)/60,decimals=2)  
    else:
        t1 = timer

    if htp != []:
        times = t1[t1<htp]
    else:
        times = t1
    
    for j in im_pd.columns:
        last_im = cv2.imread(os.path.join(image_folder, im_pd.iloc[adj,j]),-1)
        filler,alpha, beta = automatic_brightness_and_contrast(last_im,clip_hist_percent=0.001)    
        print('gel'+str(j))
        video = cv2.VideoWriter(video_name+str(j)+'.mp4', 0, 3, frameSize=(width,height),fps=5)
        for i in im_pd.index[:len(times)]:        
            im2write = cv2.imread(os.path.join(image_folder, im_pd.iloc[i,j]),-1)
            im2write = cv2.convertScaleAbs(im2write, alpha=alpha, beta=beta)
            im2write = cv2.putText(im2write,str(times[i])+' hrs',(10,20),font, fontScale, color, thickness, cv2.LINE_AA, False)
            
            imrect = cv2.rectangle(im2write,(int(800/2),int(925/2)),(int(963/2),int(950/2)),(255,255,255),-1)
            video.write(imrect)   
        
        video.release()
    
    cv2.destroyAllWindows()