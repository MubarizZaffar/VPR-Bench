#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:34:18 2020

@author: mubariz
"""

import caffe
import numpy as np
import os

def compute_map_features(ref_map):
    mean_npy = np.load(str(os.path.abspath(os.curdir))+'/VPR_Techniques/HybridNet/hybridnet_mean.npy')
    print(('Mean Array Shape:' + str(mean_npy.shape)))
    net = caffe.Net(str(os.path.abspath(os.curdir))+'/VPR_Techniques/HybridNet/deploy.prototxt',str(os.path.abspath(os.curdir))+'/VPR_Techniques/HybridNet/HybridNet.caffemodel', caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    print((net.blobs['data'].data.shape))
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean_npy)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    ref_features=[]
    features_ref_local=np.zeros((256,30))
    
    for image_reference in ref_map:
        image_reference = image_reference / 255.
        image_reference = image_reference[:,:,(2,1,0)]
        
        features_ref_local=np.zeros((256,30))

        if(image_reference is not None):
            
            transformed_image_ref = transformer.preprocess('data', image_reference)
            net.blobs['data'].data[...] = transformed_image_ref.copy()
            out = net.forward()
            features_ref=np.asarray(net.blobs['conv5'].data)[1,:,:,:].copy()
            
            for i in range(256):
    
                #S=1
                features_ref_local[i,0]=np.max(features_ref[i,:,:])
                
                #S=2
                
                features_ref_local[i,1]=np.max(features_ref[i,0:6,0:6])            
                features_ref_local[i,2]=np.max(features_ref[i,0:6,7:12])
                features_ref_local[i,3]=np.max(features_ref[i,7:12,0:6])
                features_ref_local[i,4]=np.max(features_ref[i,7:12,7:12])
                
                #S=3
                
                features_ref_local[i,5]=np.max(features_ref[i,0:4,0:4])
                features_ref_local[i,6]=np.max(features_ref[i,0:4,5:8])
                features_ref_local[i,7]=np.max(features_ref[i,0:4,9:12])
                features_ref_local[i,8]=np.max(features_ref[i,5:8,0:4])
                features_ref_local[i,9]=np.max(features_ref[i,5:8,5:8])
                features_ref_local[i,10]=np.max(features_ref[i,5:8,9:12])
                features_ref_local[i,11]=np.max(features_ref[i,9:12,0:4])
                features_ref_local[i,12]=np.max(features_ref[i,9:12,5:8])
                features_ref_local[i,13]=np.max(features_ref[i,9:12,9:12])
    
                #S=4
                features_ref_local[i,14]=np.max(features_ref[i,0:3,0:3])
                features_ref_local[i,15]=np.max(features_ref[i,0:3,4:6])
                features_ref_local[i,16]=np.max(features_ref[i,0:3,7:9])
                features_ref_local[i,17]=np.max(features_ref[i,0:3,10:12])
                features_ref_local[i,18]=np.max(features_ref[i,4:6,0:3])
                features_ref_local[i,19]=np.max(features_ref[i,4:6,4:6])
                features_ref_local[i,20]=np.max(features_ref[i,4:6,7:9])
                features_ref_local[i,21]=np.max(features_ref[i,4:6,10:12])
                features_ref_local[i,22]=np.max(features_ref[i,7:9,0:3])
                features_ref_local[i,23]=np.max(features_ref[i,7:9,4:6])
                features_ref_local[i,24]=np.max(features_ref[i,7:9,7:9])
                features_ref_local[i,25]=np.max(features_ref[i,7:9,10:12])
                features_ref_local[i,26]=np.max(features_ref[i,10:12,0:3])
                features_ref_local[i,27]=np.max(features_ref[i,10:12,4:6])
                features_ref_local[i,28]=np.max(features_ref[i,10:12,7:9])
                features_ref_local[i,29]=np.max(features_ref[i,10:12,10:12])
    
#            print(features_ref_local)
            ref_features.append(features_ref_local)
    print('Reference images descriptors computed!')    
    return ref_features

def compute_query_desc(image_query):
    
    mean_npy = np.load(str(os.path.abspath(os.curdir))+'/VPR_Techniques/HybridNet/hybridnet_mean.npy')
    print(('Mean Array Shape:' + str(mean_npy.shape)))
    net = caffe.Net(str(os.path.abspath(os.curdir))+'/VPR_Techniques/HybridNet/deploy.prototxt',str(os.path.abspath(os.curdir))+'/VPR_Techniques/HybridNet/HybridNet.caffemodel', caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    print((net.blobs['data'].data.shape))
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean_npy)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    features_query_local=np.zeros((256,30))
    

    image_query = image_query / 255.
    image_query = image_query[:,:,(2,1,0)]        
    
    if (image_query is not None):
        
        transformed_image_query = transformer.preprocess('data', image_query)
        net.blobs['data'].data[...] = transformed_image_query.copy()
        out = net.forward()
        features_query=np.asarray(net.blobs['conv5'].data)[1,:,:,:].copy()
    
        features_query_local=np.zeros((256,30))
        
        for i in range(256):
        
                #S=1
                features_query_local[i,0]=np.max(features_query[i,:,:])
                
                #S=2
                    
                features_query_local[i,1]=np.max(features_query[i,0:6,0:6])
                features_query_local[i,2]=np.max(features_query[i,0:6,7:12])
                features_query_local[i,3]=np.max(features_query[i,7:12,0:6])
                features_query_local[i,4]=np.max(features_query[i,7:12,7:12])
                
                #S=3
    
                features_query_local[i,5]=np.max(features_query[i,0:4,0:4])
                features_query_local[i,6]=np.max(features_query[i,0:4,5:8])
                features_query_local[i,7]=np.max(features_query[i,0:4,9:12])
                features_query_local[i,8]=np.max(features_query[i,5:8,0:4])
                features_query_local[i,9]=np.max(features_query[i,5:8,5:8])
                features_query_local[i,10]=np.max(features_query[i,5:8,9:12])
                features_query_local[i,11]=np.max(features_query[i,9:12,0:4])
                features_query_local[i,12]=np.max(features_query[i,9:12,5:8])
                features_query_local[i,13]=np.max(features_query[i,9:12,9:12])
    
                #S=4
                features_query_local[i,14]=np.max(features_query[i,0:3,0:3])
                features_query_local[i,15]=np.max(features_query[i,0:3,4:6])
                features_query_local[i,16]=np.max(features_query[i,0:3,7:9])
                features_query_local[i,17]=np.max(features_query[i,0:3,10:12])
                features_query_local[i,18]=np.max(features_query[i,4:6,0:3])
                features_query_local[i,19]=np.max(features_query[i,4:6,4:6])
                features_query_local[i,20]=np.max(features_query[i,4:6,7:9])
                features_query_local[i,21]=np.max(features_query[i,4:6,10:12])
                features_query_local[i,22]=np.max(features_query[i,7:9,0:3])
                features_query_local[i,23]=np.max(features_query[i,7:9,4:6])
                features_query_local[i,24]=np.max(features_query[i,7:9,7:9])
                features_query_local[i,25]=np.max(features_query[i,7:9,10:12])
                features_query_local[i,26]=np.max(features_query[i,10:12,0:3])
                features_query_local[i,27]=np.max(features_query[i,10:12,4:6])
                features_query_local[i,28]=np.max(features_query[i,10:12,7:9])
                features_query_local[i,29]=np.max(features_query[i,10:12,10:12])
                
    return features_query_local

def perform_VPR(features_query_local,ref_map_features):  

    total_Ref_Images=len(ref_map_features)
    confusion_vector=np.zeros(total_Ref_Images)
#        print(features_query_local)
    for j in range(total_Ref_Images):
 
        match_score=1-(np.sum(abs(np.subtract(features_query_local,ref_map_features[j])))/(256*256))
#            match_score=np.sum(np.dot(features_query_local,ref_map_features[j].T))/(256*256)
        confusion_vector[j]=match_score  
#    print(np.amax(confusion_vector), np.argmax(confusion_vector))
    return np.amax(confusion_vector), np.argmax(confusion_vector), confusion_vector
        

        
