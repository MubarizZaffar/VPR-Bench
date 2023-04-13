#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Tue Mar 24 21:36:57 2020

This file only shows the required format for creating ground_truth.npy file. If your ground_truth is in a CSV file, you can modify this code using python csv_reader and such.
@author: mubariz
"""

import numpy as np
total_query_images=172
total_ref_images=172

dataset_directory='/media/VPR_datasets/CampusLoop/'
save_dir=dataset_directory+'ground_truth_new.npy'

gt=np.zeros([total_query_images,2])# First column is query image index, second column is list of indices of correct matches for the query images
refs=list(range(total_ref_images))

for query in range(total_query_images):
    gt[query][0]=int(query) 
    gt[query][1]=list(range(int(max(0,query-1)),int(min(total_ref_images,query+1))))  #List of all the correct matches for query image: 'query'    

print(gt)    
np.save(save_dir,gt)    
