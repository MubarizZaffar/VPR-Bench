#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Tue Mar 24 20:35:07 2020

This file uses groundtruth information for the dataset, as imported from the file 'ground_truth_new.npy' available for each dataset. 
See the accompanying file 'numpy_gt_create.py' to get an understanding of the required ground_truth numpy file and to create new groundtruth numpy files for other datasets. 
@author: mubariz
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
import time
import math
from datetime import datetime
from create_datasetspecific_VPRperformance_summary import create_datasetspecific_VPR_summary

def compute_precision_recall(matches,scores_all):
    precision, recall, _ = precision_recall_curve(matches, scores_all)   
    return precision, recall

def compute_roc(matches,scores_all):
    fpr,tpr, _=roc_curve(matches,scores_all)
    return fpr, tpr

def compute_auc_ROC(matches,scores_all):
    if (np.sum(matches)==len(matches)): #All images are true-positives in the dataset
        print('Only single class for ROC! Computation not possible.')
        return -1
    else:
        return roc_auc_score(matches,scores_all) #This throws an error when all images are true-positives because then it is a single class only.

def compute_auc_PR(prec,recall): #Area-under-the-Precision-Recall-Curve
    return auc(recall, prec)

def compute_pcu(precision,recall,encoding_times, vpr_technique): #Performance-per-Compute-Unit
    PCU=precision[np.argwhere(recall==1)] * np.log10((np.amax(encoding_times.values())/encoding_times[vpr_technique]) + 9)
    return PCU

def compute_accuracy(matches):
    accuracy=float(np.sum(matches))/float(len(matches))
    return accuracy

def distance_vs_localisation(matches,inter_frame_distance=1):
#    distance_range=range(1,len(matches)+1)
    distance_btw_matches=[]
    prev_match_index=1
    curr_match_index=0
    for itr, match in enumerate(matches):
        curr_match_index=itr+1
        
        if (itr==0 and match==1):
            prev_match_index=curr_match_index
        
        else:
            
            if (match==1):
                distance_btw_matches.append((curr_match_index-prev_match_index)*inter_frame_distance)
                prev_match_index=curr_match_index
    
    distance_btw_matches_array=np.asarray(distance_btw_matches).reshape(len(distance_btw_matches))
    bins=range(max(distance_btw_matches_array)+1)
    distance_btw_matches_hist=np.bincount(distance_btw_matches_array) 

    return [distance_btw_matches_hist,bins]

def compute_RecallRateAtN_forRange(all_retrievedindices_scores_allqueries, ground_truth_info):
    sampleNpoints=range(1,20) #Can be changed to range(1,0.1*len(all_retrievedindices_scores_allqueries[0])) for maximum N equal to 10% of the total reference images
    recallrate_values=np.zeros(len(sampleNpoints))
    itr=0
    for N in sampleNpoints:      
        recallrate_values[itr]=compute_RecallRateAtN(N, all_retrievedindices_scores_allqueries, ground_truth_info)
        itr=itr+1
    
    print(recallrate_values)
    return recallrate_values, sampleNpoints

def compute_RecallRateAtN(N, all_retrievedindices_scores_allqueries, ground_truth_info):
    matches=[]
    total_queries=len(all_retrievedindices_scores_allqueries)
    match_found=0
    
    for query in range(total_queries):
        top_N_retrieved_ind=np.argpartition(all_retrievedindices_scores_allqueries[query], -1*N)[-1*N:]
        for retr in top_N_retrieved_ind:        
            if (retr in ground_truth_info[query][1]):
                match_found=1
                break

        if (match_found==1):
            matches.append(1)
            match_found=0
        else:
            matches.append(0)            
            match_found=0
     
    recallrate_at_N=float(np.sum(matches))/float(total_queries)
    
    return recallrate_at_N

def compute_extendedprecision(precision, recall):
    precision_at_recall_zero=precision[-2]   # This should be precision[-2] which represents the precision at lowest recall. Because the value of preicision[-1] is always 1 and doesn't have a corresponding threhold. See scikit-learn PR-Curves documentation.
    if (precision_at_recall_zero<1):
        recall_at_precision_one=0
    else: 
        recall_at_precision_one=recall[min(np.argwhere(precision==1))]
    
    extended_precision=(precision_at_recall_zero + recall_at_precision_one)/2
  
    return extended_precision

def compute_RMF(encoding_time, matching_time, matches, platform_Velocity=10,  camera_fps=50, distance_based_sampling_fpm=1, scaling_factor=1): # This has been depreciated and replaced by compute_RMF()# No. of prospective place matching (or loop-closure) candidates based on retriveal time and accuracy
     # platform_Velocity: Platform Velocity is taken as 1 meters per second, for example.
     # camera_fps: FPS sampling of the onboard camera
    
     # distance_based_sampling_fpm: By default use 1 frames every 1 meter
     # scaling_factor: Downsampling of incoming frames based on any criterion
    original_total_matches=np.sum(matches) 
    realtime_total_matches=0 # Unlike in any other metric, the matches here reflect the correct place matches by a technique given its real-world retrieval speed  
    incoming_framerate=min(math.ceil(scaling_factor*(distance_based_sampling_fpm)*platform_Velocity), camera_fps) #The ideal VPR_retrieval_fps should be equal to or higher than this value
    
    no_of_query_images=len(matches)
    search_cost=1*no_of_query_images   #(1) is the worst case linear search, you can change this to logarithmic etc. depending on your search mechanism.
    vpr_retrieval_fps=(1.0/(encoding_time+(matching_time*search_cost)))
    incomingrate_to_retrievalrate_ratio=int(max(float(incoming_framerate)/float(vpr_retrieval_fps),1))   
  
    for itr, match in enumerate(matches):
        
        if (((itr+1)%incomingrate_to_retrievalrate_ratio)==0 or itr==0):
            if (match==1):
                realtime_total_matches=realtime_total_matches+1
    
    return original_total_matches,realtime_total_matches

def save_results(tech,dataset,retrieved,matches,scores_all,encoding_time,matching_time,auc_pr,auc_roc,pcu,ep,accuracy,recallrate_at_K_range, sampleNpoints, ori_TM, rt_TM):    
    ts = time.time()
    ts_datetime='' # Replace '' with str(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')) if you would like to save files with timestamp
    
    with open('VPR_Bench_Results/'+tech+'_'+dataset+'-VPRBench'+ts_datetime+'.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    
        for i in range(len(retrieved)):   
            row=str(i) + ',' + str(retrieved[i]) + ',' +  str(matches[i]) + ',' +  str(scores_all[i])
            my_writer.writerow([row])
            print(row)
        
        row='EncodingTime: ' + str(encoding_time) + ',' + 'MatchingTime: ' + str(matching_time) + ',' +  'AUC-PR: ' + str(auc_pr) + ','+  'AUC-ROC: ' + str(auc_roc) + ',' +  'PCU: ' + str(pcu) + ',' +  'EP: ' + str(ep)+ ',' +  'Accuracy: ' + str(accuracy)+ ',' + 'recall_rate_at_K_range: ' +  str(recallrate_at_K_range) + ',' + 'recallrate_K_range: ' + str(sampleNpoints) + ',' + 'Original Total Matches: ' + str(ori_TM) + ',' + 'Realtime Total Matches: ' + str(rt_TM) 
        my_writer.writerow([row])
    csvfile.close()   

def draw_RecallRateAtKCurves(recallrate_at_K_dict,sampleNpoints,vpr_techniques,dataset):
    plt.figure()
    for tech in vpr_techniques:
    
        plt.plot(sampleNpoints, recallrate_at_K_dict[tech], label=tech.replace("_Precomputed",""))
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('N',fontsize='x-large')
        plt.ylabel('RecallRate',fontsize='x-large')
    
    plt.title(dataset)
    plt.grid()     
    plt.savefig('RecallRateCurves/'+dataset+'-RecallRateCurves'+'.png') 
#    plt.show()     

def draw_ROC_Curves(fpr_dict,tpr_dict,techniques,dataset):
    plt.figure()
    for tech in techniques:
    
        plt.plot(fpr_dict[tech], tpr_dict[tech], label=tech.replace("_Precomputed",""))
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('FPR',fontsize='x-large')
        plt.ylabel('TPR',fontsize='x-large')
    
    plt.title(dataset)
    plt.grid()     
    plt.savefig('ROCCurves/'+dataset+'-ROCcurves'+'.png') 
#    plt.show()    
     
def draw_PR_Curves(prec_dict,recall_dict,techniques,dataset):   
    plt.figure()
    for tech in techniques:
    
        plt.plot(recall_dict[tech], prec_dict[tech], label=tech.replace("_Precomputed",""))
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('Recall',fontsize='x-large')
        plt.ylabel('Precision',fontsize='x-large')

    plt.title(dataset)    
    plt.grid()     
    plt.savefig('PRCurves/'+dataset+'-PRcurves'+'.png') 
#    plt.show()           

def draw_retrievalfps_vs_platformspeed(encoding_times, matching_times, vpr_techniques, const_distance=2): # Where const_distance=1 means that 1 frame must be available every 2 meters
    print(encoding_times)
    print(matching_times)
    
    platform_velocity=np.arange(0,100,5)
    camera_fps=(1.0/const_distance)*platform_velocity   #These are the values of camera FPS required to have a frame at const_distance given platform_velocity. 
    print('camera_fps', camera_fps)
    no_of_map_images=np.array([1,10,100,500,1000,5000])
    print('no_of_map_images',no_of_map_images)
    search_costs=1*no_of_map_images   #(1) is the worst case linear search, you can change this to logarithmic etc. depending on your search mechanism.
    
    VPR_fps={}
    for vpr_technique in vpr_techniques:
        print(vpr_technique)
        vpr_fps_differentsearch=np.zeros(len(search_costs))        
        for itr,map_search_cost in enumerate(search_costs):
            vpr_fps_differentsearch[itr]=(1.0/(encoding_times[vpr_technique]+(matching_times[vpr_technique]*map_search_cost)))

        print(vpr_fps_differentsearch)
        VPR_fps[vpr_technique]=vpr_fps_differentsearch
        
    print('VPR_fps',VPR_fps)
    
    fig,axs=plt.subplots(1,len(vpr_techniques),figsize=(30,3),squeeze=False)
    for itr,tech in enumerate(vpr_techniques):
        
        axs[0,itr].plot(platform_velocity,camera_fps, label='FPS_Req')
        for i, vpr_fps_diffmapsize in enumerate(VPR_fps[tech]):    
            axs[0,itr].plot(platform_velocity, np.repeat(vpr_fps_diffmapsize,len(platform_velocity)) ,label='FPS_VPR='+str('%.1f'%vpr_fps_diffmapsize)+' at Z='+str(int(no_of_map_images[i])))
#        axs[0,itr].set_xticks(viewpoint_positions)
        axs[0,itr].set(xlabel='Platform Speed (mps)', ylabel='Frames-Per-Second')
        axs[0,itr].title.set_text(tech.replace("_Precomputed",""))
        axs[0,itr].legend(loc="upper right")
    fig.tight_layout()    
    fig.savefig('VPR_FPS_vs_PlatformSpeed_PDF.png')

def draw_retrieval_vs_distance(distance_btw_matches_info, techniques, dataset):

    plt.figure()
    for tech in techniques:
    
        plt.plot(distance_btw_matches_info[tech][1], distance_btw_matches_info[tech][0], label=tech.replace("_Precomputed",""), drawstyle='steps')
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('Inter-Frame Distance (meters)',fontsize='x-large')
        plt.ylabel('No. of Consecutive Correct Matches',fontsize='x-large')    
    
    plt.title(dataset)    
    plt.grid()
    plt.savefig('Retrieval_vs_Distance_Curves/'+dataset+'-Retrieval_vs_Distance_Curves'+'.png') 
#    plt.show() 
        
def compute_matches(retrieved_all, ground_truth_info):
    matches=[]
    itr=0
    for retr in retrieved_all:
        if (retr in ground_truth_info[itr][1]):
            matches.append(1)
        else:
            matches.append(0)
        itr=itr+1        
    return matches

def performance_comparison(dataset_name, dataset_directory,vpr_techniques,query_all, retrieved_all, scores_all, encoding_time_all, matching_time_all, all_retrievedindices_scores_allqueries_dict,descriptor_shape_dict): #dataset_directory has a numpy file named ground_truth_new.npy
    ground_truth_info=np.load(dataset_directory+'ground_truth_new.npy',allow_pickle=True) # A 2-dimensional array representing the range of reference image matches correponding to a query image   
    prec_dict={}
    recall_dict={}
    auc_pr_dict={}
    fpr_dict={}
    tpr_dict={}
    auc_roc_dict={}
    pcu_dict={}
    ep_dict={}
    accuracy_dict={}
    RMF_dict={}    
    recallrate_at_K_dict={}
    sampleNpoints=[]
    distance_btw_matches_hist_dict={}
                        
    for vpr_technique in vpr_techniques:          
        print(vpr_technique)
        matches=compute_matches(retrieved_all[vpr_technique], ground_truth_info)
        prec,recall=compute_precision_recall(matches,scores_all[vpr_technique])
        auc_pr=compute_auc_PR(prec,recall)
        fpr,tpr=compute_roc(matches,scores_all[vpr_technique])
        auc_roc=compute_auc_ROC(matches,scores_all[vpr_technique])
        pcu=compute_pcu(prec,recall,encoding_time_all, vpr_technique)
        recallrate_at_K_range, sampleNpoints=compute_RecallRateAtN_forRange(all_retrievedindices_scores_allqueries_dict[vpr_technique], ground_truth_info) #K range is 1 to 20
        ep=compute_extendedprecision(prec, recall)
        accuracy=compute_accuracy(matches)
        ori_TM,rt_TM=compute_RMF(encoding_time_all[vpr_technique], matching_time_all[vpr_technique], matches)
        distance_btw_matches_hist=distance_vs_localisation(matches)
        save_results(vpr_technique,dataset_name,retrieved_all[vpr_technique],matches,scores_all[vpr_technique],encoding_time_all[vpr_technique], matching_time_all[vpr_technique],auc_pr,auc_roc,pcu,ep,accuracy,recallrate_at_K_range, sampleNpoints, ori_TM, rt_TM)

        prec_dict[vpr_technique]=prec     #values in prec correspond to 1->0 recall in recall
        recall_dict[vpr_technique]=recall #values vary from 1->0 with increasing index    
        auc_pr_dict[vpr_technique]=auc_pr
        fpr_dict[vpr_technique]=fpr
        tpr_dict[vpr_technique]=tpr
        auc_roc_dict[vpr_technique]=auc_roc
        pcu_dict[vpr_technique]=pcu
        ep_dict[vpr_technique]=ep
        accuracy_dict[vpr_technique]=accuracy
        RMF_dict[vpr_technique]=[ori_TM, rt_TM]
        recallrate_at_K_dict[vpr_technique]=recallrate_at_K_range
        distance_btw_matches_hist_dict[vpr_technique]=distance_btw_matches_hist

    create_datasetspecific_VPR_summary(vpr_techniques, dataset_name, auc_pr_dict, auc_roc_dict, pcu_dict, ep_dict, encoding_time_all, matching_time_all, RMF_dict, descriptor_shape_dict)
    draw_retrievalfps_vs_platformspeed(encoding_time_all, matching_time_all, vpr_techniques)
    draw_PR_Curves(prec_dict,recall_dict,vpr_techniques,dataset_name) 
    draw_ROC_Curves(fpr_dict,tpr_dict,vpr_techniques,dataset_name) 
    draw_RecallRateAtKCurves(recallrate_at_K_dict,sampleNpoints,vpr_techniques,dataset_name)    
    draw_retrieval_vs_distance(distance_btw_matches_hist_dict,vpr_techniques,dataset_name)

    print('AUC PR:-',auc_pr_dict)
    print('AUC ROC:-',auc_roc_dict)
    print('PCU:-',pcu_dict)
    print('EP:-',ep_dict)
    print('Encoding Time:-',encoding_time_all)
    print('Matching Time:-',matching_time_all)
    print('RMF_dict',RMF_dict)
    print('AUC_ROC_dict',auc_roc_dict)    
    print('recallrate_at_K_dict', recallrate_at_K_dict)  
    print('descriptor_shape_dict', descriptor_shape_dict)    

