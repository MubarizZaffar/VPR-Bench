#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:52:21 2019

@author: mubariz
"""

import cv2
import time
import numpy as np

image = cv2.imread("/media/mubariz/New Volume/VPR_datasets/ESSEX3IN1_dataset/query_combined/0.jpg")
image = cv2.resize((image), (512,512))

winSize = (32,32)
blockSize = (32,32)
blockStride = (32,32)
cellSize = (16,16)
nbins = 8
#derivAperture = 1
#winSigma = 4.
#histogramNormType = 0
#L2HysThreshold = 2.0000000000000001e-01
#gammaCorrection = 0
#nlevels = 64

start=time.time()
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#winStride = (8,8)
#padding = (0,0)
#locations = ((10,20),)
hist = hog.compute(image,winStride,padding,locations)
#newhist=np.reshape(hist,[1024,8])
print('Time',time.time()-start)
print(len(hist))
#print(len(newhist),len(newhist[0]))