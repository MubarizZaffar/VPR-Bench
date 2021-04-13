#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:28:55 2020

Just so you know how exactly the dataset images are named in our paper.

@author: mubariz
"""
import os
import cv2
import natsort
import glob

query_dir='/home/mubariz/Documents/VPR_Bench Datasets/Pittsburgh/query/'
ref_dir='/home/mubariz/Documents/VPR_Bench Datasets/Pittsburgh/ref/'

query_dir_new='/home/mubariz/Documents/VPR_Bench Datasets/Pittsburgh/query_new/'
ref_dir_new='/home/mubariz/Documents/VPR_Bench Datasets/Pittsburgh/ref_new/'

if not os.path.exists(query_dir_new):
    os.makedirs(query_dir_new)
if not os.path.exists(ref_dir_new):
    os.makedirs(ref_dir_new)
  
query_images_names=[os.path.basename(x) for x in glob.glob(query_dir+'*.jpg')]
ref_images_names=[os.path.basename(x) for x in glob.glob(ref_dir+'*.jpg')] 

itr=0
for query in natsort.natsorted(query_images_names,reverse=False):
    print('Reading Image Named: '+query_dir+query)
    img=cv2.imread(query_dir+query)
    cv2.imwrite(query_dir_new+str(itr).zfill(7)+'.jpg',img)
    itr=itr+1

itr=0
for ref in natsort.natsorted(ref_images_names,reverse=False):
    print('Reading Image Named: '+ref_dir+ref)
    img=cv2.imread(ref_dir+ref)
    cv2.imwrite(ref_dir_new+str(itr).zfill(7)+'.jpg',img)
    itr=itr+1
        

