# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:45:46 2021

@author: schulmanlab
"""
import numpy as np
from skimage.transform import resize
import skimage.draw as skdraw
from pycromanager import Bridge
import time
import matplotlib.pyplot as plt
from skimage.draw import polygon
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


bridge = Bridge(convert_camel_case=False)
core = bridge.get_core()
DMD = core.getSLMDevice()

#%% Initialization and Micromanager functions:
def init():    
    bridge = Bridge(convert_camel_case=False)
    core = bridge.get_core()
    DMD = core.getSLMDevice()
    core.setProperty(DMD,'TriggerType',1)
    core.setProperty('UserDefinedStateDevice-1','Label','Patterning ON (dichroic mirror)')
    core.setProperty('UserDefinedStateDevice','Label','BF')
    core.setProperty('UserDefinedShutter-1','State',1)
    core.setProperty('UserDefinedShutter','State',1)    
    #h = 684, w = 608
    #Channel 4: UV LED
    core.setProperty('Mightex_BLS(USB)','mode','NORMAL')
    core.setProperty(DMD,'AffineTransform.m00',-0.6625)
    core.setProperty(DMD,'AffineTransform.m01',0.0000)
    core.setProperty(DMD,'AffineTransform.m02',1078.5840)
    core.setProperty(DMD,'AffineTransform.m10',0.0000)
    core.setProperty(DMD,'AffineTransform.m11',-1045.9139)
    #current set: 0-1000
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0)
    
def position_list():
    '''Returns numpy position list extracted from Micromanager.'''
    mm = bridge.get_studio()
    pm = mm.positions()
    pos_list = pm.getPositionList()
    numpos = pos_list.getNumberOfPositions()
    np_list = np.zeros((numpos,2))
    for idx in range(numpos):
        pos = pos_list.getPosition(idx)
        stage_pos = pos.get(0)
        np_list[idx,0] = stage_pos.x
        np_list[idx,1] = stage_pos.y          
    return np_list
    
#%% Flow controller functions:    
d = {'s1': '00001', 's2': '00010','s3': '000100', 's4': '01000', 's5': '10000'}
d = {'s0': 0, 's1': 1, 's2': 2,'s3': 4, 's4': 8, 's5': 16}
df = pd.Series(data=d)
  
def valve_on(switch):
    """Turn Solenoid valve on: Input must be something like 's0', 's1', etc."""   
    core.setProperty('Arduino-Switch','State',int(df.get(switch)))

def valve_off(switch2='s0'):
    """Turn Solenoid valve off: No input needed.""" 
    core.setProperty('Arduino-Switch','State',int(df.get(switch2)))
    
def valve_timer(switch, wait):
    """Turn Solenoid valve on: Input 1 must be something like 's0', 's1', etc., input 2 must be time in seconds"""
    # Switch must be something like 's0', 's1', etc.
    core.setProperty('Arduino-Switch','State',int(df.get(switch)))
    for m in range(0, wait):
        time.sleep(1)
    valve_off()

#%%Mask Generator Functions:
def circle_mask_generator(h,w,radius,cx=0,cy=0):  
    """Returns binary mask with a circle in the center for use with a DMD.
    h,w: height, width of mask.
    radius: radius of circle in the center of the mask."""
    rr,cc = skdraw.disk((h/2+cx,w/2+cy),radius,shape=[h,w])
    mask1 = np.zeros([h,w],dtype='uint8')
    mask1[rr,cc] = 255
    return mask1
  
def equil_triangle_mask_generator(h,w,base):
    """Returns equilateral triangle mask with a triangle in the center for use with a DMD.
    h,w: height, width of mask.
    base: length of base in the center of the mask."""
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
    return (mask1)
        
def square_mask_generator(h,w,ex):
    """Returns square mask with a square in the center for use with a DMD.
    h,w: height, width of mask.
    ex: length of any side in the center of the mask."""
    rr,cc = skdraw.rectangle(((h-ex)/2,(w-ex)/2),extent=(ex,ex),shape=[h,w])
    mask2 = np.zeros((h,w),dtype='uint8')
    mask2[rr.astype('int'),cc.astype('int')] = 255
    return mask2
    
def rectangle_mask_generator(h,w,lx,ly,cx=0,cy=0):
    """Returns rectangular mask with a rectangle in the center for use with a DMD.
    h,w: height, width of mask.
    lx,ly: length,width of rectangle.
    cx,cy: x and y offset from center of mask"""
    midx = h/2
    midy = w/2
    lx = lx/2
    ly = ly/2
    startx = midx-lx
    starty = midy-ly
    endx = midx + lx
    endy = midy + ly
    rr,cc = skdraw.rectangle((startx+cx, starty+cy),end=(endx+cx, endy+cy),shape=[h,w])
    mask2 = np.zeros((h,w),dtype='uint8')
    mask2[rr.astype('int'),cc.astype('int')] = 255
    return mask2

def hollow_rr_mask_generator(h,w,lxl,lyl,lxs,lys,cxl=0,cyl=0,cxs=0,cys=0):
    """Returns rectangular mask with a rectangular hole in the center for use with a DMD.
    h,w: height, width of mask.
    lxl,lyl: length,width of large rectangle.
    lxs,lys: length,width of small rectanglar hole
    cxl,cyl: center offset for larger rectangle
    cxs,cys: center offset for smaller rectangle"""
    llarge = rectangle_mask_generator(h, w, lxl, lyl,cx=cxl,cy=cyl)
    lsmall = rectangle_mask_generator(h, w, lxs, lys,cx=cxs,cy=cys)    
    lcomb = llarge - lsmall
    return lcomb

def hollow_rc_mask_generator(h,w,lxl,lyl,radius):
    """Returns rectangular mask with a rectangular hole in the center for use with a DMD.
    h,w: height, width of mask.
    lxl,lyl: length,width of large rectangle.
    radius: radius of small circular hole"""
    lrect = rectangle_mask_generator(h, w, lxl, lyl)
    lsmall = circle_mask_generator(h, w, radius)    
    lcomb = lrect - lsmall
    return lcomb

def plus_mask_generator(h,w,wdist = 90, wthick = 30):
    """Returns a plus-shaped mask with a plus in the center for use with a DMD.
    h,w: height, width of mask.
    wdist: length (from end to end) of the plus
    wthick: thickness of each member of the plus"""
    m1 = rectangle_mask_generator(h,w,wthick,wdist)
    m2 = rectangle_mask_generator(h,w,wdist,wthick)
    return m1 + m2

def message_mask_generator(h,w,message="ABC",fontsize = 100,fonttype = "ariblk.ttf"):
    """Returns a mask with a message within it for use with a DMD.
    h,w  height, width of mask.
    character: any standard character [A-Z, 0-9]
    fontsize: size of the font, in pixels
    fonttype: any font located within the font directory on the OS."""
    img = Image.new('1', (h,w), color = 'black')
    font = ImageFont.truetype(fonttype, fontsize)
    # font = ImageFont.truetype("arial.ttf", fontsize)

    d = ImageDraw.Draw(img)
    dx,dy = d.textsize(message,font=font)
    d.text(((h-dx)/2,(w-dy)/2), message,font=font,stroke_fill = 50 , fill=(255),align="left") #TO ALIGN CHARACTER IN CENTER
    img=np.pad(img,pad_width=0, mode='constant', constant_values=0) #Manually added padding
    img=Image.fromarray(img)
    npimg = np.array(img)
    toR = np.zeros(npimg.shape,dtype='uint8')
    toR[npimg] = 255
    return toR

def mask_rescaler(h,w,in1):
    """Rescaling function - REQUIRED for uploading a mask to DMD.
    h,w: height, width of mask.
    in1: mask to rescale. Returns rescaled mask"""
    y1 = resize(in1,(h,w/2))
    wpad = int(w/4)
    ypad = np.pad(y1,((0,0),(wpad,wpad)),'constant', constant_values=(0))
    ypad=np.array(ypad,dtype='uint8')
    ypad[ypad==1]=255
    return ypad

#%%Patterning functions:
def patterning(UVexposure,slimage,channel=4,intensity=1000):
    '''Exposes an image by turning on the UV LED for a set amount of time.
    UVexposure: exposure time (in seconds)
    slimage: rescaled mask uploaded.
    channel: UV/Blue LED
    intensity: LED intensity'''
    core.setSLMImage(DMD,slimage)
    time.sleep(1.5)
    core.setProperty('Mightex_BLS(USB)','channel',channel)
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',intensity)
    time.sleep(UVexposure)
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0)
    time.sleep(1)
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0)
    
def matrix_patterner(mat1,exposure,slimage,stage,valveon = [],dist = 60,ch=4,inte=1000):
    '''Patterns a grid of hydrogels at location specified in binary matrix mat1.
    mat1: locations of each hydrogel in grid.
    exposure: UV exposure for all hydrogels.
    slimage: hydrogel shape as a uint8 image.
    valveon: turning a particular valve to pattern.
    dist: distance between each hydrogel
    channel: UV/Blue LED
    intensity: LED intensity'''
    x=core.getXPosition()
    y=core.getYPosition()
    for i in range(len(mat1)):
        for j in range(len(mat1)):
            if mat1[i,j]:
                core.setXYPosition(x,y)
                time.sleep(3)
                core.setRelativeXYPosition((dist)*i,(dist)*j)   
                time.sleep(2)
                if not valveon:
                    valve_timer(valveon, 5) #flows pregel 5 seconds
                patterning(exposure,slimage,channel=ch,intensity=inte)
                time.sleep(.5)

                
def ploc_patterner(arr1,gridlen,exposure,radii=[],valveon = [],dist = 60,ch=4,inte=1000):
    '''Patterns a set of cylindrical hydrogels at fractions (from 0 to 1) specified in 2xn array arr1.
    if radii = 0, then radii is default 50 um. 
    the total size of the 
    arr1: locations of each hydrogel relative to grid size
    exposure: UV exposure for all hydrogels.
    slimage: hydrogel shape.
    valveon: turning a particular valve to pattern.
    dist: distance between each hydrogel
    channel: UV/Blue LED
    intensity: LED intensity
    the percent location patterner uses the top left corner as reference starting position.'''
    h = 684
    w = 608
    CF = 0.45

    x=core.getXPosition()
    y=core.getYPosition()
    arrdist = arr1*gridlen
    for i in range(np.size(arr1,1)):
        core.setXYPosition(x,y)

        if radii == []:
            diam = 100
            diam_conv = diam / CF
            draw_circle = circle_mask_generator(h,w,radius=(diam_conv / 2))
        else:
            rad = radii[i]
            rad_conv = rad/CF
            draw_circle = circle_mask_generator(h,w,radius=(rad_conv))
            
        time.sleep(1.5)
        core.setRelativeXYPosition(arrdist[0,i],arrdist[1,i])   
        patterning(exposure,draw_circle,channel=ch,intensity=inte)
        time.sleep(.5)           


#%%Imaging functions:
    
    
#%%Functions that need more work:
def grid_pattern_v2(dist,exposure,slimage,numgels=3, ch=4,inte = 1000):
    '''Patterns a grid of hydrogels at all locations. specified in binary matrix mat1.
    mat1: locations of each hydrogel in grid.
    exposure: UV exposure for all hydrogels.
    slimage: hydrogel shape.
    valveon: turning a particular valve to pattern.
    dist: distance between each hydrogel
    channel: UV/Blue LED
    intensity: LED intensity'''
    x=core.getXPosition()
    y=core.getYPosition()
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
                
                
def rect_pattern_v2(rx,exposure,numgels=6,dist=150,ch=4,inte=1000):
    x=core.getXPosition()
    y=core.getYPosition()
    buf =(numgels-1)*10
    yh = rx/6+20
    hnum = int(numgels/2)
    xspace = np.linspace(x-rx/2*hnum-buf,x+rx/2*hnum+buf,num=numgels)
    x2=xspace[:hnum]-dist/2
    x3 = xspace[hnum:]+dist/2
    xt = np.concatenate((x2,x3))
    
    yspace = np.linspace(y-yh-buf,y+yh+buf,3)
    xv,yv = np.meshgrid(xt,yspace,indexing='ij')
    slimage = mask_rescaler(square_mask_generator(ex=[rx/6,rx]))
    for i in range(len(xspace)):
        for j in range(3):
            core.setXYPosition(xv[i,j],yv[i,j])
            patterning(exposure,slimage,channel=ch,intensity=inte)
            time.sleep(2)  
            
def rect_pattern_v3(h,w,exposure,numgels=6,dist=75,ch=4,inte=1000):
    x=core.getXPosition()
    y=core.getYPosition()
    rx = 75/0.45*3
    xs2 = np.linspace(0,300*numgels,numgels)+dist
    # xs2 = np.arange(0,rx*numgels,rx/2)
    x2r = x+xs2
    x2l = x-xs2
    xtot = np.concatenate((x2l,x2r))
    slimage = mask_rescaler(h,w,square_mask_generator(h,w,ex=[rx/6,rx]))
    for j in [-1,1]:
        # core.setXYPosition(x,y)
        core.sleep(1)
        core.setXYPosition(x,y)
        time.sleep(3)
        core.setRelativeXYPosition((dist+330)/2*j,0)   
        patterning(exposure,slimage,channel=ch,intensity=inte)
        time.sleep(2)
        for i in range(1,numgels):
            # core.setXYPosition(xtot[i],y)
            core.setRelativeXYPosition(330*j,0)   
            patterning(exposure,slimage,channel=ch,intensity=inte)
            time.sleep(2)