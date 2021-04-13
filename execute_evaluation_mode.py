#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Thu Mar  5 12:05:56 2020

@author: mubariz
"""

from evaluate_vpr_techniques import evaluate_vpr_techniques
from performance_comparison import performance_comparison
from perform_vpr_invariance_analysis import perform_vpr_invariance_analysis, perform_vpr_viewpointinvariance_analysis_validation, perform_vpr_illuminationinvariance_analysis_validation      
        
def exec_eval_mode(VPR_evaluation_mode, dataset_name, dataset_directory,precomputed_directory,VPR_techniques, save_descriptors, scale_percent):
        
    if (VPR_evaluation_mode==0): #Evaluate VPR techniques on a given dataset for AUC, PCU, EP, RMF, RecallRate and others.
        print('Evaluation Mode 0')
        query_all, retrieved_all, scores_all, encoding_time_all, matching_time_all, all_retrievedindices_scores_allqueries_dict, descriptor_shape_dict=evaluate_vpr_techniques(dataset_directory,precomputed_directory,VPR_techniques,save_descriptors, scale_percent)   #Evaluates all VPR techniques currently available in the framework on specified dataset. 
        performance_comparison(dataset_name, dataset_directory, VPR_techniques,query_all, retrieved_all, scores_all, encoding_time_all, matching_time_all, all_retrievedindices_scores_allqueries_dict, descriptor_shape_dict)

    elif (VPR_evaluation_mode==1): #Invariance analysis of VPR techniques on Point Feature dataset
        print('Evaluation Mode 1')
        variation_quantified_dataset_directory='variation_quantified_datasets/Pointfeatures_dataset/'# The path to the PointFeatures dataset for VPR Evaluation Mode 1.
        perform_vpr_invariance_analysis(variation_quantified_dataset_directory,VPR_techniques)   #Evaluates all VPR techniques currently available in the framework on Point Features dataset. 

    elif (VPR_evaluation_mode==2): #Viewpoint Invariance analysis validation of VPR techniques on QUT Multi-lane dataset
        print('Evaluation Mode 2')
        variation_quantified_dataset_directory='variation_quantified_datasets/QUT_Multilane_dataset/'# The path to the QUT Multilane dataset for VPR Evaluation Mode 2. 
        perform_vpr_viewpointinvariance_analysis_validation(variation_quantified_dataset_directory,VPR_techniques)   #Evaluates all VPR techniques currently available in the framework on QUT multi-lane dataset. 

    elif (VPR_evaluation_mode==3): #Illumination Invariance analysis Validation of VPR techniques on MIT Multi-illumination dataset
        print('Evaluation Mode 3')
        variation_quantified_dataset_directory='variation_quantified_datasets/MIT_Multiillumination_dataset/' # The path to the MIT multi-illumination dataset for VPR Evaluation Mode 3.
        perform_vpr_illuminationinvariance_analysis_validation(variation_quantified_dataset_directory,VPR_techniques)   #Evaluates all VPR techniques currently available in the framework on MIT multi-illumination dataset. 
