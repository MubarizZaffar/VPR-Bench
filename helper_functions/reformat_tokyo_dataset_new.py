#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:34:01 2021

@author: mubariz
"""

'''

This code takes the original Tokyo 24/7 dataset as input, and outputs
a VPR-Bench compatible version of the same dataset.

'''

import numpy as np
from scipy.spatial.distance import cdist
import cv2
import os

#########################################################################
# Please set the paths below accordingly #
original_gt_filepath='/media/Tokyo247_full_dataset/tokyo247_dbStruct.npz'
original_querydb_path='/media/Tokyo247_full_dataset/247query_subset_v2/'  # This folder contains 315 query images from the original link of Torii et al.
original_refdb_path='/media/Tokyo247_full_dataset/247_database/' # This folder contains 16 sub-folders (03814-03829) of ref images from the original link of Torii et al.

new_gt_filepath='/media/Tokyo247_full_dataset_reformatted_78k/ground_truth_new_78k_1603212249.npy'
new_querydb_path='/media/Tokyo247_full_dataset_reformatted_78k/query/'
new_refdb_path='/media/Tokyo247_full_dataset_reformatted_78k/ref/'
           
#########################################################################
if not os.path.exists(new_querydb_path):
    os.makedirs(new_querydb_path)
if not os.path.exists(new_refdb_path):
    os.makedirs(new_refdb_path)
    
db=np.load(original_gt_filepath)
utmDb=db['utmDb']
utmQ=db['utmQ']

dMat = cdist(utmDb,utmQ)
bestMatchPerQuery = np.argmin(dMat,axis=0)
positives_within_25_meters = np.array([np.argwhere(dMat[:,qIdx] <= 25).flatten() for qIdx in range(dMat.shape[1])])
ref_dict={}
ref_itr=0
total_queries=len(db['qImage'])
print((list(db.keys())))
total_ref=len(db['dbImage'])
print(total_ref)
gt_new=np.zeros((total_queries,2), dtype=np.ndarray)

for itr in range(total_queries):
    print(itr)
    query_image_path=original_querydb_path+str(db['qImage'][itr])
    query=cv2.imread(query_image_path)
    cv2.imwrite(new_querydb_path+str(itr)+'.jpg',query)

    for itr2 in range(len(positives_within_25_meters[itr])):
        key=original_refdb_path+str(db['dbImage'][positives_within_25_meters[itr][itr2]]).split('.jpg')[0]+'.png'
        print(key)

        if key not in ref_dict:
            
            ref=cv2.imread(key)
            if (ref is None):
                print(('ref image not found. Path: ', key))
            else:
                print('Write image')
                ref_dict[key]=ref_itr
                cv2.imwrite(new_refdb_path+str(ref_itr)+'.jpg',ref)
                ref_itr=ref_itr+1

        else:
            print('Key exists!!!!')   

print(ref_itr)
print(ref_dict)

#########Adding Distractor Images ############################################

for itr in range (total_ref):
    key=original_refdb_path+str(db['dbImage'][itr]).split('.jpg')[0]+'.png'
    if key not in ref_dict:
        ref=cv2.imread(key)
        if (ref is None):
            print(('ref image not found. Path: ', key))
        else:
            print('Write image')
            ref_dict[key]=ref_itr
            cv2.imwrite(new_refdb_path+str(ref_itr)+'.jpg',ref)
            ref_itr=ref_itr+1        
            
###########Creating Ground-truth##############################################

for itr in range(total_queries):
    print(itr)
    my_list=[]
    for itr2 in range(len(positives_within_25_meters[itr])):
        key=original_refdb_path+str(db['dbImage'][positives_within_25_meters[itr][itr2]]).split('.jpg')[0]+'.png'
        ref=cv2.imread(key)
        
        if (ref is None):
            print(('ref image not found. Path: ', key))
        else:
            print('Ground-truth Added')
            my_list.append(ref_dict[key])

    gt_new[itr][0]=itr
    gt_new[itr][1]=my_list

print(gt_new)
np.save(new_gt_filepath,gt_new) 