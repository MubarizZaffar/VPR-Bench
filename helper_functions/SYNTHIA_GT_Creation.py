#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:06:58 2020

@author: mubariz
"""
import os
import numpy as np
import math
input_dir_query='/media/SYNTHIA-SEQS-04-NIGHT/CameraParams/Stereo_Left/Omni_F/'
input_dir_ref='/media/SYNTHIA-SEQS-04-FALL/CameraParams/Stereo_Left/Omni_F/'

newgt_path='/media/VPRDatasets/Synthia-NightToFall/ground_truth_new.npy'
q_itr=0
gt_new=np.zeros((813,2), dtype=np.ndarray)

for filename in sorted(os.listdir(input_dir_query)):
    print(filename)
    gt=open(input_dir_query+'/'+filename,'r')
    gt_pose_4x4_q=gt.read()

    query_x_value=float(gt_pose_4x4_q.split(' ')[-4])
    query_z_value=float(gt_pose_4x4_q.split(' ')[-2])
    
    ref_itr=0
    gt_list=[]  
    
    for ref_filename in sorted(os.listdir(input_dir_ref)):

        gt_ref=open(input_dir_ref+'/'+ref_filename,'r')
        gt_pose_4x4_r=gt_ref.read()
    
        ref_x_value=float(gt_pose_4x4_r.split(' ')[-4])
        ref_z_value=float(gt_pose_4x4_r.split(' ')[-2])
        
        diff_x=ref_x_value-query_x_value
        diff_z=ref_z_value-query_z_value
        diff_r=int(math.sqrt(diff_x*diff_x + diff_z*diff_z))
        
        if (diff_r<7):
            gt_list.append(ref_itr)
        
        ref_itr=ref_itr+1
    
    gt_new[q_itr][0]=q_itr
    gt_new[q_itr][1]=gt_list
    
    q_itr=q_itr+1
    
print(gt_new)
np.save(newgt_path,gt_new)

gt_new_read=np.load(newgt_path)
for i in range(len(gt_new_read)):
    print(gt_new_read[i])
