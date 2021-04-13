import caffe
import numpy as np
import cv2
from time import time
import os
from os.path import dirname

first_it = True
A = None

def compute_map_features(ref_map_images):
    ref_map_images_features=[]
    
    alexnet_proto_path=os.path.join(dirname(__file__),"alexnet/deploy.prototxt")
    alexnet_weights=os.path.join(dirname(__file__),"alexnet/bvlc_alexnet.caffemodel")
    alexnet = caffe.Net(alexnet_proto_path,1,weights=alexnet_weights)

    transformer_alex = caffe.io.Transformer({'data':(1,3,227,227)})    
    transformer_alex.set_raw_scale('data',1./255)
    transformer_alex.set_transpose('data', (2,0,1))
    transformer_alex.set_channel_swap('data', (2,1,0))
    
    for ref_img in ref_map_images:
            img_yuv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            alex_conv3 = None      
    
            im2 = cv2.resize(im, (227,227), interpolation=cv2.INTER_CUBIC)
            alexnet.blobs['data'].data[...] = transformer_alex.preprocess('data', im2)
            alexnet.forward()
            alex_conv3 = np.copy(alexnet.blobs['conv3'].data[...])
            alex_conv3 = np.reshape(alex_conv3, (alex_conv3.size, 1))
            global first_it
            global A
            if first_it:
                np.random.seed(0)
                A = np.random.randn(1064, alex_conv3.size) # For Gaussian random projection  descr[0].size=1064
                first_it = False
            alex_conv3 = np.matmul(A, alex_conv3)
            alex_conv3 = np.reshape(alex_conv3, (1, alex_conv3.size))
            alex_conv3 /= np.linalg.norm(alex_conv3)

            ref_map_images_features.append(alex_conv3)
    
    return ref_map_images_features

def compute_query_desc(image_query):

    alexnet_proto_path=os.path.join(dirname(__file__),"alexnet/deploy.prototxt")
    alexnet_weights=os.path.join(dirname(__file__),"alexnet/bvlc_alexnet.caffemodel")
    alexnet = caffe.Net(alexnet_proto_path,1,weights=alexnet_weights)

    transformer_alex = caffe.io.Transformer({'data':(1,3,227,227)})    
    transformer_alex.set_raw_scale('data',1./255)
    transformer_alex.set_transpose('data', (2,0,1))
    transformer_alex.set_channel_swap('data', (2,1,0))

    img_yuv = cv2.cvtColor(image_query, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    alex_conv3 = None      

    im2 = cv2.resize(im, (227,227), interpolation=cv2.INTER_CUBIC)
    alexnet.blobs['data'].data[...] = transformer_alex.preprocess('data', im2)
    alexnet.forward()
    alex_conv3 = np.copy(alexnet.blobs['conv3'].data[...])
    alex_conv3 = np.reshape(alex_conv3, (alex_conv3.size, 1))
    global first_it
    global A
    if first_it:
        np.random.seed(0)
        A = np.random.randn(1064, alex_conv3.size) # For Gaussian random projection  descr[0].size=1064
        first_it = False
    alex_conv3 = np.matmul(A, alex_conv3)
    alex_conv3 = np.reshape(alex_conv3, (1, alex_conv3.size))
    alex_conv3 /= np.linalg.norm(alex_conv3)
    
    return alex_conv3

def perform_VPR(alex_conv3,ref_map_features):
    
    matching_scores=[]
    
    for ref_desc in ref_map_features:
        score=np.dot(alex_conv3,ref_desc.T)
        matching_scores.append(score)

    return np.amax(matching_scores), np.argmax(matching_scores), np.asarray(matching_scores).reshape(len(ref_map_features)) 








