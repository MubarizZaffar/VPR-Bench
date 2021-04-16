#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Mon Feb 17 13:35:46 2020

Please modify this file as per your VPR system. We have provided a usual interface required for a VPR system so any VPR system can be plugged-in here. 
Implement the functions 'compute_map_features', 'compute_query_desc', 'perform_VPR' in your desired VPR technique according to the example provided in the file CoHOG_Python/CoHOG.py. 
Don't change the interface here to maintain the stack clean. A range of VPR techniques have been already implemented in our work, see below.

@author: mubariz
"""
import time
import sys
    
def selective_import(VPR_technique):

#The below imported functions need to be implemented in your VPR system if you are integrating a new VPR technique. Please modify your VPR technique code accordingly and set the import paths here. This is the only change required to integrate a new VPR technique, except your prospective dependencies.    
    
    if (VPR_technique=='CoHOG'):
        from VPR_Techniques.CoHOG_Python.CoHOG import compute_map_features, compute_query_desc, perform_VPR
    
    elif (VPR_technique=='AMOSNet'):
        from VPR_Techniques.AMOSNet import compute_map_features, compute_query_desc, perform_VPR
    
    elif (VPR_technique=='HybridNet'):
        from VPR_Techniques.HybridNet import compute_map_features, compute_query_desc, perform_VPR
        
    elif (VPR_technique=='CALC'):
        from VPR_Techniques.CALC import compute_map_features, compute_query_desc, perform_VPR
    
    elif (VPR_technique=='NetVLAD'):
        from VPR_Techniques.NetVLAD.NetVLAD import compute_map_features, compute_query_desc, perform_VPR
        
    elif (VPR_technique=='RegionVLAD'):
        from VPR_Techniques.RegionVLAD.RegionVLAD import compute_map_features, compute_query_desc, perform_VPR
        
    elif (VPR_technique=='AlexNet_VPR'):
        from VPR_Techniques.AlexNet_VPR.AlexNet_VPR import compute_map_features, compute_query_desc, perform_VPR
        
    elif (VPR_technique=='HOG'):
        from VPR_Techniques.HOG_VPR import compute_map_features, compute_query_desc, perform_VPR

    else:
        sys.exit("Method {} not supported. Please check if letters' case match exactly.".format(VPR_technique))
   
    return compute_map_features, compute_query_desc, perform_VPR     

def compute_image_descriptors(robot_map, vpr_tech='CoHOG'): #Takes in a list of reference images as outputs a list of feature descriptors corresponding to these images. 
    compute_map_features, compute_query_desc, perform_VPR = selective_import(vpr_tech) #Imports the VPR template functions for the specified 'vpr_tech'
    map_features=compute_map_features(robot_map)
#    print(vpr_tech + ' Descriptor Size: ',np.asarray(map_features[0]).shape)
#    print(vpr_tech + ' Descriptor Type: ',np.asarray(map_features[0]).dtype)
    return map_features

def match_two_images(query_image,ref,vpr_tech='CoHOG'): #For matching two images only.
    compute_map_features, compute_query_desc, perform_VPR = selective_import(vpr_tech) #Imports the VPR template functions for the specified 'vpr_tech'
    ref_desc=compute_map_features(ref)
    query_desc=compute_query_desc(query_image)
    matching_score,matched_vertex,_=perform_VPR(query_desc,ref_desc) 
    
    return matching_score,matched_vertex

def place_match(query_image,robot_map_features,vpr_tech='CoHOG'): #For matching an input query image with a precomputed map of reference descriptors.
    compute_map_features, compute_query_desc, perform_VPR = selective_import(vpr_tech) #Imports the VPR template functions for the specified 'vpr_tech'
    t1=time.time()
    query_desc=compute_query_desc(query_image)
    t2=time.time()
    matching_score,matched_vertex, confusion_vector=perform_VPR(query_desc,robot_map_features)  
    t3=time.time()
    
    return 1,matched_vertex,matching_score,(t2-t1),((t3-t2)/len(robot_map_features)), confusion_vector