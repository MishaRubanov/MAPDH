# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:21:32 2022

@author: rmish
"""
import os
import image_processor_v5 as IP
numgels =12

image_folder = 'D:\\2.6.22_rO1_v5_gradient_37C_repeat_v2\\6_CounterSpike'
images_FAM = [img for img in os.listdir(image_folder) if img.endswith(".tiff")]
# im_pd_FAM,t1 = IP.imSorter(images_FAM,numgels,'')
IP.renamer(image_folder,images_FAM,newt=111860)
