#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Sun Oct 11 03:58:46 2020

@author: mubariz

The results of VPR techniques on each dataset are scattered across their individual CSV files (in subfolder 'VPR_Bench_Results') and numpy files (in sub-folder 'precomputed_matches').
Therefore, this code creates a CSV for each dataset that summarises the performance (AUC-PR, AUC-ROC, EP, RMF, RecallRate, Encoding Time, Matching Time) of all VPR techniques and these
summaries are available in the sub-folder 'dataset_specific_VPR_summary'.

"""

import csv

def create_datasetspecific_VPR_summary(techniques, dataset, auc_pr_dict, auc_roc_dict, pcu_dict, ep_dict, encoding_time_all, matching_time_all, RMF_dict, descriptor_shape_dict) :
    
    with open('dataset_specific_VPR_summary/'+dataset.replace('/','-')+'.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row='Technique Names,'
        for tech in techniques:
            row=row+tech+','
        my_writer.writerow([row])
        
        row='AUC-PR,'
        for tech in techniques:
            row=row+str(auc_pr_dict[tech])+','
        my_writer.writerow([row])

        row='AUC-ROC,'
        for tech in techniques:
            row=row+str(auc_roc_dict[tech])+','
        my_writer.writerow([row])
        
        row='EP,'
        for tech in techniques:
            row=row+str(ep_dict[tech])+','
        my_writer.writerow([row])
        
        row='Encoding Time,'
        for tech in techniques:
            row=row+str(encoding_time_all[tech])+','
        my_writer.writerow([row])
        
        row='Matching Time,'
        for tech in techniques:
            row=row+str(matching_time_all[tech])+','
        my_writer.writerow([row])
        
        row='PCU,'
        for tech in techniques:
            row=row+str(pcu_dict[tech])+','
        my_writer.writerow([row])
        
        row='TMF,'
        for tech in techniques:
            row=row+str(RMF_dict[tech][0])+','
        my_writer.writerow([row])
        
        row='RMF,'
        for tech in techniques:
            row=row+str(RMF_dict[tech][1])+','
        my_writer.writerow([row])

#        row='Descriptor Size,'
#        for tech in techniques:
#            row=row+descriptor_shape_dict[tech]+','
#        my_writer.writerow([row])
        
    csvfile.close()  