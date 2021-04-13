#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Sat Feb 15 16:03:24 2020

This is the main file for VPR-Bench. Here you can set most parameters that you will need
for performance evaluation. Under normal scenarios, this is the only file that you would
need to execute (using ``python main.py'' in terminal).
"""

from execute_evaluation_mode import exec_eval_mode

dataset_name='Corridor' # This string is used when creating titles for plots in VPR Evaluation Mode 0, so please specify the dataset name here if you are using Mode 0.
vpr_dataset_directory='datasets/corridor/' #The path to a particular dataset you want to use for VPR Evaluation Mode 0. The folder 'datasets' is a sub-folder within the VPR-Bench folder that contains query images, ref images and ground_truth.npy.
vpr_precomputed_matches_directory='precomputed_matches/corridor/' # The path to storing or reading precomputed matching information for VPR Evaluation Mode 0. The folder 'precomputed_matches' is a sub-folder within the VPR-Bench folder. See below text for further understanding.

"""
The above 'vpr_precomputed_matches_directory' is used when you already have the matching information 
for some techniques on some datasets. This could happen in 3 scenarios.

1. The precomputed matching information for all techniques and datasets used in our IJCV paper is 
   provided in the 'precomputed_matches' sub-folder and you want to use some of it.
   
2. You ran VPR_Bench Evaluation Mode 0 on some new dataset and you had set 'save_matching_info=1' which
   would have saved this matching information into 'vpr_precomputed_matches_directory'.

3. You had computed matching information using another platform and you want to integrate it into 
   VPR-Bench so you can create PR-Curves and others with all the legends in a single plot.

Important Note: 'vpr_precomputed_matches_directory' is also used for saving matching information when 
you have set 'save_matching_info=1', so make sure you have specified this path for saving matching info.
"""

VPR_techniques=['CoHOG','CALC'] # Currently available options are ['NetVLAD','RegionVLAD','CoHOG','HOG','AlexNet_VPR','AMOSNet','HybridNet','CALC']. 

"""
Some examples of setting 'VPR_techniques' are:
    1. VPR_techniques=['NetVLAD','CoHOG','AlexNet_VPR','HybridNet','CALC'] if you only want to use a sub-set of the 8 VPR techniques.
    2. VPR_techniques=['NetVLAD_Precomputed','RegionVLAD_Precomputed','CoHOG_Precomputed','HOG_Precomputed','AlexNet_VPR_Precomputed','AMOSNet_Precomputed','HybridNet_Precomputed','CALC_Precomputed','ap-gem-r101_Precomputed','denseVLAD_Precomputed'] if you only want to use precomputed matching info of all the VPR techniques.
    3. VPR_techniques=['NetVLAD_Precomputed','CoHOG','AlexNet_VPR_Precomputed','HybridNet'] if you want to use precomputed matching info for NetVLAD and AlexNet but you do not have this matching info for CoHOG and HybridNet. 

Basically, append '_Precomputed' to a technique's name if you want to use its precomputed matching info (instead of recomputing the whole thing again) available in 'vpr_precomputed_matches_directory'.
"""

VPR_evaluation_mode=0 # See below few lines for explanation.

"""
VPR_evaluation_mode=0 for dataset-based (e.g. Tokyo 24/7 dataset, Gardens Point dataset and the 10 others in our benchmark) VPR performance evaluation. 
VPR_evaluation_mode=1 for viewpoint and illumination invariance analysis of VPR techniques on Point Features dataset.
VPR_evaluation_mode=2 for viewpoint invariance analysis of VPR techniques on QUT multi-lane dataset.
VPR_evaluation_mode=3 for illumination invariance analysis of VPR techniques on MIT multi-illumination dataset.

So only one mode at a time. For VPR_evaluation_mode 1/2/3 just set 'VPR_techniques' to your choice of techniques, as these modes don't use any other parameter of this main file.
"""

save_matching_info=1 # If save_matching_info=0, save matching info in 'vpr_precomputed_matches_directory' for all the techniques in 'VPR_techniques' on the dataset specified in 'vpr_dataset_directory'.
scale_percent=100 # Provision for resizing (with aspect-ratio maintained) of query and reference images between 0-100%. 100% is equivalent to NO resizing.
    
def main():     
   exec_eval_mode(VPR_evaluation_mode, dataset_name, vpr_dataset_directory,vpr_precomputed_matches_directory, VPR_techniques, save_matching_info, scale_percent)
     
if __name__ == "__main__":
    main()