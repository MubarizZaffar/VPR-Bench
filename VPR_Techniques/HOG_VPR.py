#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:49:42 2020

@author: mubariz
"""
import cv2
import numpy as np
import time

def compute_map_features(ref_map):  #ref_map is a 1D list of images in this case.
    
    winSize = (512,512)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (16,16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    ref_desc_list=[]
    
    for ref_image in ref_map:
        
        if ref_image is not None:    
            hog_desc=hog.compute(cv2.resize(ref_image, winSize))
            
        ref_desc_list.append(hog_desc)
        print(hog_desc)
        
    return ref_desc_list

def compute_query_desc(query):
        
    winSize = (512,512)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (16,16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    query_desc=hog.compute(cv2.resize(query, winSize))
    
    print(query_desc)
    print(query_desc.shape)
    
    return query_desc

def perform_VPR(query_desc,ref_map_features): #ref_map_features is a 1D list of feature descriptors of reference images in this case.

    confusion_vector=np.zeros(len(ref_map_features))
    itr=0
    for ref_desc in ref_map_features:
        t1=time.time()
        query_desc=query_desc.astype('float64')
        ref_desc=ref_desc.astype('float64')
        score=np.dot(query_desc.T,ref_desc)/(np.linalg.norm(query_desc)*np.linalg.norm(ref_desc))
        t2=time.time()
        print('HOG tm:',t2-t1)
        confusion_vector[itr]=score
        itr=itr+1
        
    return np.amax(confusion_vector), np.argmax(confusion_vector), confusion_vector