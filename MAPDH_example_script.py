# -*- coding: utf-8 -*-
"""
This script gives examples for how to pattern different shapes within MAPDH.
"""

#%%Initialization
#import relevant functions
import time
import patterning_functions as PF
from pycromanager import Bridge
import numpy as np
import cv2

#Creating a link with Micromanager.
core = Bridge.get_core()
DMD = core.getSLMDevice()

#Initializing hardware
PF.init()

#Extracting the position list
pos_list = PF.position_list()

#Height and width of the DMD
h = core.getSLMHeight(DMD)
w = core.getSLMWidth(DMD)

#Conversion Factor: converting between pixels and microns patterned
#1 pixel = 0.45 um
#For example, to pattern a 100um gel, the diameter should be 100/0.45 = 222 pixels
CF = 0.45

#Set exposure time for patterning
UVexposure = 0.5

#%%Generating masks

# Circle Mask: 100 um diameter
diam = 100
diam_conv = diam / CF
draw_circle = PF.circle_mask_generator(h,w,radius=(diam_conv / 2))
draw_circle_rescaled = PF.mask_rescaler(draw_circle) #Important step: scaling mask to DMD

# Square Mask: 50 um side
square_side = 50
square_conv = square_side / CF
draw_square = PF.square_mask_generator(h,w,ex=square_conv)
draw_square_rescaled = PF.mask_rescaler(draw_square) #Important step: scaling mask to DMD

# Equilateral triangle Mask: 100 um base
base_side = 100
base_conv = base_side / CF
draw_triangle = PF.equil_triangle_mask_generator(h, w, base=base_conv)
draw_triangle_rescaled = PF.mask_rescaler(draw_triangle) #Important step: scaling mask to DMD

# Rectangle Mask: 100x50 um
lx_side = 100
ly_side = 50
lx_conv = lx_side / CF
ly_conv = ly_side / CF
draw_rectangle = PF.rectangle_mask_generator(h, w, lx=lx_conv, ly=ly_conv)
draw_rectangle_rescaled = PF.mask_rescaler(draw_rectangle)

# Plus Mask: each bar is 100 um wide and 40 um thick
width_side = 100/2
thick_side = 40/2
width_conv = width_side / CF
thick_conv = thick_side / CF
draw_plus = PF.plus_mask_generator(h, w, wdist=width_conv, wthick=thick_conv)
draw_plus_rescaled = PF.mask_rescaler(draw_plus)

# Bullseye mask: combining two different masks
# inner circle diameter: 50, outer circle diameter: 100
diam1 = 50
diam2 = 100
in_diam_conv = diam1 / CF
in_circle = PF.circle_mask_generator(h,w,radius=(in_diam_conv / 2))
out_diam_conv = diam2 / CF
out_circle = PF.circle_mask_generator(h,w,radius=(out_diam_conv / 2))
bullseye = out_circle-in_circle
bullseye_rescaled = PF.mask_rescaler(bullseye)

allmasks_rescaled = [draw_circle_rescaled, draw_square_rescaled, draw_triangle_rescaled, draw_plus_rescaled, bullseye_rescaled]

#Valve labels: s0-s4: pregel inks. s5: wash solution
valves = ('s0', 's1', 's2', 's3', 's4', 's5')

#Imaging settings
fluorophore = ("skip", 'Cy3','GFP-FAM','Cy5', 'BF') #Fluorophore to use
light_pillar_intensity = 0.3 # for patterning off/Empty
LED_brightness = 100 #XCite lamp intensity in %
imaging_exposure = (0, 400, 400, 400, 10) #fluorescence exposure in ms
filename = 'MAPDH_ex_'
fluoro_label = ("skip", 'Cy3gels_', 'Tye665gels_','ATTO488gels_','AllColorgels_')


for stage in range (0,4):
    
    #%%Patterning: 5 different hydrogels are patterned with four different inks
    PF.valve_timer(valves[stage], 90) #flows pregel 90 seconds 
    
    for i in range(0,5): #pattern four hydrogels
        core.setXYPosition(pos_list[i,0],pos_list[i,1]) #move to first location    
        core.setRelativeXYPosition(0, (stage-1)*150.0) #move to different location based on stage           

        time.sleep(3) #wait at least 1 second for microscope to reach location
        PF.patterning(UVexposure,allmasks_rescaled[i],channel=4,intensity=1000)         
        PF.valve_timer(valves[stage], 5) #flows pregel 5 seconds 
    
    core.setProperty('Mightex_BLS(USB)','normal_CurrentSet',0) #Redundant UV LED turn-off command
    PF.valve_timer(valves[5], 60 ) #flows wash solution for 60 seconds after patterning is over.
    print('Wash ended, beginning imaging')  

    #%%Imaging: 4 different channels
    core.setProperty('UserDefinedStateDevice-1','Label','Empty')
    core.setProperty('UserDefinedShutter-1','State',1)
    core.setProperty('UserDefinedShutter','State',1)
    
    for i in range(len(fluorophore)):
        if fluorophore[i] == "skip":
            continue
        if fluorophore[i] == "BF":
            core.setProperty('DTOL-DAC-0', 'Volts', light_pillar_intensity)
            core.setProperty('XCite-Exacte','Lamp-Intensity', 10)
        else:
            core.setProperty('DTOL-DAC-0', 'Volts', 0)
            core.setProperty('XCite-Exacte','Lamp-Intensity', LED_brightness)
        core.setProperty('UserDefinedStateDevice','Label',fluorophore[i])
        core.setExposure(imaging_exposure[i])
        time.sleep(3)
        core.snapImage()
        tagged_image = core.getTaggedImage()
        pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
       
        cv2.imwrite(filename + str(fluoro_label[stage]) + '_' + str(fluorophore[i])+'.tif', pixels.astype(np.uint16) )
        core.setProperty('DTOL-DAC-0', 'Volts', 0)
        time.sleep(2) 