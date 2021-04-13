import caffe
import numpy as np
import cv2
import os

def computeForwardPasses(net, im, transformer, resize_net):
    """
    Compute the forward passes for CALC
    """

    img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if not resize_net:
        im = cv2.resize(im, (160, 120), interpolation = cv2.INTER_CUBIC)
    else:
        transformer = caffe.io.Transformer({'X1':(1,1,im.shape[0],im.shape[1])})    
        transformer.set_raw_scale('X1',1./255)
        
        x1 = net.blobs['X1']
        x1.reshape(1,1,im.shape[0],im.shape[1])
        net.reshape()

    net.blobs['X1'].data[...] = transformer.preprocess('X1', im)
    net.forward()
    d = np.copy(net.blobs['descriptor'].data[...])
    d /= np.linalg.norm(d)
        
#    print('desc[0].size:',descr[0].size)

    return d

def compute_map_features(ref_map_images):
    
    net_def_path=str(os.path.abspath(os.curdir))+'/VPR_Techniques/CALC/proto/deploy.prototxt'
    net_model_path=str(os.path.abspath(os.curdir))+'/VPR_Techniques/CALC/model/calc.caffemodel'
    resize_net=False 
    caffe.set_mode_cpu()
    # caffe.set_device(0)
               
    net=caffe.Net(net_def_path,1,weights=net_model_path)

    database = [] # stored pic descriptors   
    
    # Use caffe's transformer
    transformer = caffe.io.Transformer({'X1':(1,1,120,160)})    
    transformer.set_raw_scale('X1',1./255)

    for img in ref_map_images:    
        
        descr= computeForwardPasses(net, img, transformer, resize_net)        
        database.append(descr)         
    
    return database

def compute_query_desc(im):
    
    net_def_path=str(os.path.abspath(os.curdir))+'/VPR_Techniques/CALC/proto/deploy.prototxt'
    net_model_path=str(os.path.abspath(os.curdir))+'/VPR_Techniques/CALC/model/calc.caffemodel'
    resize_net=False 
    caffe.set_mode_cpu()
    # caffe.set_device(0)

    net=caffe.Net(net_def_path,1,weights=net_model_path)

    # Use caffe's transformer
    transformer = caffe.io.Transformer({'X1':(1,1,120,160)})    
    transformer.set_raw_scale('X1',1./255)
    
    descr= computeForwardPasses(net, im, transformer, resize_net) 
    
    return descr

def perform_VPR(descr,database):       

    all_scores=[]

    for i in range(len(database)):
        query_desc=descr.astype('float64')
        ref_desc=database[i].astype('float64')
        curr_sim = np.dot(query_desc, ref_desc.T) # Normalizd vectors means that this give cosine similarity
        all_scores.append(curr_sim)

    return np.amax(all_scores), np.argmax(all_scores), np.asarray(all_scores).reshape(len(database)) 
