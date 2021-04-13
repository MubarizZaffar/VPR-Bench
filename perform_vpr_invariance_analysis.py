#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:08:00 2020

@author: mubariz
"""
import numpy as np
import cv2
from vpr_system import match_two_images
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import pickle

def compute_area_between_the_curves(h_axis,match_curve,mismatch_curve):
    
    polygon_points = [] #creates a empty list where we will append the points to create the polygon
    
    for itr in range(len(match_curve)):
        polygon_points.append([h_axis[itr],match_curve[itr]]) #append all xy points for curve 1
    
    for itr in range(len(mismatch_curve)):
        polygon_points.append([h_axis[itr],mismatch_curve[itr]]) #append all xy points for curve 2 in the reverse order (from last point to first point)
    
    polygon_points.append([h_axis[0],match_curve[0]]) #append the first point in curve 1 again, to it "closes" the polygon
    
    polygon = Polygon(polygon_points)
    area = polygon.area
    print(area)
    
    return area

def perform_vpr_invariance_analysis(variation_quantified_dataset_directory,VPR_techniques):
    results_already_exist=False # Toggle this if you have previously executed this function and results are available in pickle file, and you don't want to recompute the results but just play around with the graphs.  

    viewpoint_variations_varied_scores_dict={} # Matching scores of same places with viewpoint variations
    viewpoint_variations_mismatch_scores_dict={} # Matching scores of different places for comparison with above
    viewpoint_variations_ABC_dict={} #ABC is area-between-the-curves
    
    illumination_variation_varied_scores_dict={} # Matching scores of same places with illumination variations
    illumination_variation_mismatch_scores_dict={} # Matching scores of different places for comparison with above
    illumination_variations_ABC_dict={}
    
    viewpoint_positions=range(119)  #Total 119 different camera positions labelled between 1-119, see paper for more details
    illumination_variations=range(19)  #Total 19 different LED illumination labelled between 1-19, see paper for more details

    if (results_already_exist==True): #If you have previously run the code for some techniques and just want to plot the results. False by default.
        f1=open('invariance_dict/viewpoint_variations_varied_scores_dict.pkl','rb')
        viewpoint_variations_varied_scores_dict = pickle.load(f1)
        f2=open('invariance_dict/viewpoint_variations_mismatch_scores_dict.pkl','rb')
        viewpoint_variations_mismatch_scores_dict = pickle.load(f2)
        f3=open('invariance_dict/illumination_variation_varied_scores_dict.pkl','rb')
        illumination_variation_varied_scores_dict = pickle.load(f3)
        f4=open('invariance_dict/illumination_variation_mismatch_scores_dict.pkl','rb')
        illumination_variation_mismatch_scores_dict = pickle.load(f4)
        f5=open('invariance_dict/viewpoint_variations_ABC_dict.pkl','rb')
        viewpoint_variations_ABC_dict = pickle.load(f5)
        f6=open('invariance_dict/illumination_variations_ABC_dict.pkl','rb')
        illumination_variations_ABC_dict = pickle.load(f6)

    else:      
        for tech in VPR_techniques:
            matching_scores_varied_with_fixed_viewpoint_and_varied_illumination=np.zeros(len(illumination_variations)) #Image at viewpoint position 1 is matched with itself and with all other different viewpoint positions
            matching_scores_varied_with_varied_viewpoint_and_fixed_illumination=np.zeros(len(viewpoint_positions)) #Image at viewpoint position 1 is matched with itself and with all other different viewpoint positions
           
            matching_scores_mismatch_with_fixed_viewpoint_and_varied_illumination=np.zeros(len(illumination_variations)) #Image at viewpoint position 1 is matched with other different places
            matching_scores_mismatch_with_varied_viewpoint_and_fixed_illumination=np.zeros(len(viewpoint_positions)) #Image at viewpoint position 1 is matched with other different places 
            
            for viewpoint in viewpoint_positions:
                fixed_illumination=0
                base_viewpoint=0
                base_image_name='Img'+str(base_viewpoint+1).zfill(3)+'_'+str(fixed_illumination+1).zfill(2)+'.bmp'  #Extra 1 is added in indices to map position '0' here to position '1' of dataset        
                base_image=cv2.imread(variation_quantified_dataset_directory+'SET001/'+base_image_name)       
                varied_image_name='Img'+str(viewpoint+1).zfill(3)+'_'+str(fixed_illumination+1).zfill(2)+'.bmp'  #Extra 1 is added to map position '0' here to position '1' of dataset
                varied_image=cv2.imread(variation_quantified_dataset_directory+'SET001/'+varied_image_name)
                mismatch_image_name='Img'+str(viewpoint+1).zfill(3)+'_'+str(fixed_illumination+1).zfill(2)+'.bmp'  #Extra 1 is added to map position '0' here to position '1' of dataset
                mismatch_image=cv2.imread(variation_quantified_dataset_directory+'SET004/'+mismatch_image_name)
                
                compatible_list_variedimage=[]
                compatible_list_variedimage.append(varied_image)
                
                compatible_list_mismatchimage=[]
                compatible_list_mismatchimage.append(mismatch_image)
           
                if (base_image is not None and varied_image is not None and mismatch_image is not None):
                    score_varied,_=match_two_images(base_image,compatible_list_variedimage,tech)
                    score_mismatch,_=match_two_images(base_image,compatible_list_mismatchimage,tech)
                    matching_scores_varied_with_varied_viewpoint_and_fixed_illumination[viewpoint]=score_varied
                    matching_scores_mismatch_with_varied_viewpoint_and_fixed_illumination[viewpoint]=score_mismatch
                else:
                    print('Images Not Found')                
                    
            for illumination in illumination_variations:
                fixed_viewpoint=0        
                base_illumination=0
                base_image_name='Img'+str(fixed_viewpoint+1).zfill(3)+'_'+str(base_illumination+1).zfill(2)+'.bmp'  #Extra 1 is added to map position '0' here to position '1' of dataset        
                base_image=cv2.imread(variation_quantified_dataset_directory+'SET001/'+base_image_name)
                varied_image_name='Img'+str(fixed_viewpoint+1).zfill(3)+'_'+str(illumination+1).zfill(2)+'.bmp'  #Extra 1 is added to map position '0' here to position '1' of dataset
                varied_image=cv2.imread(variation_quantified_dataset_directory+'SET001/'+varied_image_name)
                mismatch_image_name='Img'+str(fixed_viewpoint+1).zfill(3)+'_'+str(illumination+1).zfill(2)+'.bmp'  #Extra 1 is added to map position '0' here to position '1' of dataset
                mismatch_image=cv2.imread(variation_quantified_dataset_directory+'SET004/'+mismatch_image_name)
                
                compatible_list_variedimage=[]
                compatible_list_variedimage.append(varied_image)
    
                compatible_list_mismatchimage=[]
                compatible_list_mismatchimage.append(mismatch_image)
    
                if (base_image is not None and varied_image is not None and mismatch_image is not None):        
                    score_varied,_=match_two_images(base_image,compatible_list_variedimage,tech)
                    score_mismatch,_=match_two_images(base_image,compatible_list_mismatchimage,tech)
                    matching_scores_varied_with_fixed_viewpoint_and_varied_illumination[illumination]=score_varied
                    matching_scores_mismatch_with_fixed_viewpoint_and_varied_illumination[illumination]=score_mismatch
                else:
                    print('Images Not Found')
            
            viewpoint_variations_varied_scores_dict[tech]=matching_scores_varied_with_varied_viewpoint_and_fixed_illumination
            illumination_variation_varied_scores_dict[tech]=matching_scores_varied_with_fixed_viewpoint_and_varied_illumination
            viewpoint_variations_mismatch_scores_dict[tech]=matching_scores_mismatch_with_varied_viewpoint_and_fixed_illumination
            illumination_variation_mismatch_scores_dict[tech]=matching_scores_mismatch_with_fixed_viewpoint_and_varied_illumination


###############################################################################################################################################
#This block of code plots the matching scores for all 8 techniques given illumination and viewpoint variations (Fig. 19 of our paper). I have left this code uncommented
#here for your convenience. You would note that it is designed for even number of VPR techniques and works best when all the 8 VPR techniques 
#are being used for analysis. You may want to change the exact arrangements of subplots for the best possible readibility for the number of techniques
#you may want to use.
    
    fig,axs=plt.subplots(4,len(VPR_techniques)/2,figsize=(12,12))
    
    row=0
    col=0
    for itr,tech in enumerate(VPR_techniques):
        abc=compute_area_between_the_curves(viewpoint_positions,viewpoint_variations_varied_scores_dict[tech],viewpoint_variations_mismatch_scores_dict[tech])
        viewpoint_variations_ABC_dict[tech]=abc
        
        if (itr<4):
            row=0
            col=itr
        else:
            row=1
            col=itr-4
            
        axs[row,col].plot(range(1,120),viewpoint_variations_varied_scores_dict[tech], label='Same Place')
        axs[row,col].plot(range(1,120),viewpoint_variations_mismatch_scores_dict[tech], label='Different Place')
#        axs[0,itr].set_xticks(viewpoint_positions)
        axs[row,col].set(xlabel='Viewpoint Position', ylabel='Matching Score')
        axs[row,col].title.set_text(tech+', ABC='+str("%0.2f"%abc))
        axs[row,col].legend(loc="upper right")
#        plt.plot(viewpoint_variations_varied_scores_dict[tech], label=tech+' Correct Match')
#        plt.plot(viewpoint_variations_mismatch_scores_dict[tech], label=tech+' False Match')
#        plt.legend()
#        plt.title('Viewpoint Varied, ABC='+str(abc))
#        plt.figure()

    row=2
    col=0
    
    for itr,tech in enumerate(VPR_techniques):
        abc=compute_area_between_the_curves(illumination_variations,illumination_variation_varied_scores_dict[tech],illumination_variation_mismatch_scores_dict[tech])
        illumination_variations_ABC_dict[tech]=abc

        if (itr<4):
            row=2
            col=itr
        else:
            row=3
            col=itr-4
            
        axs[row,col].plot(range(1,20),illumination_variation_varied_scores_dict[tech], label='Same Place')
        axs[row,col].plot(range(1,20),illumination_variation_mismatch_scores_dict[tech], label='Different Place')
#        axs[0,itr].set_xticks(viewpoint_positions)
        axs[row,col].set(xlabel='Illumination State', ylabel='Matching Score')
        axs[row,col].title.set_text(tech+', ABC='+str("%0.2f"%abc))
        axs[row,col].legend(loc="upper right")
#        plt.plot(illumination_variation_varied_scores_dict[tech], label=tech+' Correct Match')
#        plt.plot(illumination_variation_mismatch_scores_dict[tech], label=tech+' False Match')
#        plt.legend()
#        plt.title('Illumination Varied, ABC='+str(abc))
#        plt.figure()

    
    fig.tight_layout()
    fig.savefig('Invariance_All_VPRBench_PDF_4x4.pdf')  #Saves the figure in the main directory of project.
    
##################################################################################################################

    
###Computing these results takes time, so it's suggested to store the results at the end of execution using below few lines of pickle storage.###

    f1 = open("invariance_dict/viewpoint_variations_varied_scores_dict.pkl","wb")
    pickle.dump(viewpoint_variations_varied_scores_dict,f1)
    f1.close()

    f2 = open("invariance_dict/viewpoint_variations_mismatch_scores_dict.pkl","wb")
    pickle.dump(viewpoint_variations_mismatch_scores_dict,f2)
    f2.close()

    f3 = open("invariance_dict/illumination_variation_varied_scores_dict.pkl","wb")
    pickle.dump(illumination_variation_varied_scores_dict,f3)
    f3.close()

    f4 = open("invariance_dict/illumination_variation_mismatch_scores_dict.pkl","wb")
    pickle.dump(illumination_variation_mismatch_scores_dict,f4)
    f4.close()

    f5 = open("invariance_dict/viewpoint_variations_ABC_dict.pkl","wb")
    pickle.dump(viewpoint_variations_ABC_dict,f5)
    f5.close()

    f6 = open("invariance_dict/illumination_variations_ABC_dict.pkl","wb")
    pickle.dump(illumination_variations_ABC_dict,f6)
    f6.close()    

def perform_vpr_viewpointinvariance_analysis_validation(variation_quantified_validation_dataset_directory,VPR_techniques):

    viewpoint_variations_varied_scores_dict={} # Matching scores of same places with viewpoint variations
    viewpoint_variations_mismatch_scores_dict={} # Matching scores of different places for comparison with above
    viewpoint_variations_ABC_dict={} #ABC is area-between-the-curves

    viewpoint_positions=range(15)  #Total 15 different camera positions labelled between 1-14, see paper for more details

    for tech in VPR_techniques:
        matching_scores_varied_with_varied_viewpoint_and_fixed_illumination=np.zeros(len(viewpoint_positions)) #Image at viewpoint position 1 is matched with itself and with all other different viewpoint positions
        matching_scores_mismatch_with_varied_viewpoint_and_fixed_illumination=np.zeros(len(viewpoint_positions)) #Image at viewpoint position 1 is matched with other different places 
        
        for viewpoint in viewpoint_positions:
            base_viewpoint=0
            base_image_name=str(base_viewpoint) + '.jpg'    
            base_image=cv2.imread(variation_quantified_validation_dataset_directory+'Place1/'+base_image_name)       
            varied_image_name=str(viewpoint)+'.jpg' 
            varied_image=cv2.imread(variation_quantified_validation_dataset_directory+'Place1/'+varied_image_name)
            mismatch_image_name=str(viewpoint)+'.jpg' 
            mismatch_image=cv2.imread(variation_quantified_validation_dataset_directory+'Place2/'+mismatch_image_name)
            
            compatible_list_variedimage=[]
            compatible_list_variedimage.append(varied_image)
            
            compatible_list_mismatchimage=[]
            compatible_list_mismatchimage.append(mismatch_image)
       
            if (base_image is not None and varied_image is not None and mismatch_image is not None):
                score_varied,_=match_two_images(base_image,compatible_list_variedimage,tech)
                score_mismatch,_=match_two_images(base_image,compatible_list_mismatchimage,tech)
                matching_scores_varied_with_varied_viewpoint_and_fixed_illumination[viewpoint]=score_varied
                matching_scores_mismatch_with_varied_viewpoint_and_fixed_illumination[viewpoint]=score_mismatch
            else:
                print('Images Not Found')                
                
        
        viewpoint_variations_varied_scores_dict[tech]=matching_scores_varied_with_varied_viewpoint_and_fixed_illumination
        viewpoint_variations_mismatch_scores_dict[tech]=matching_scores_mismatch_with_varied_viewpoint_and_fixed_illumination


###############################################################################################################################################
#This block of code plots the matching scores for all 8 techniques given viewpoint variations of QUT Multi-lane dataset (Fig. 20 of our paper). I have left this code uncommented
#here for your convenience. You would note that it is designed for even number of VPR techniques and works best when all the 8 VPR techniques 
#are being used for analysis. You may want to change the exact arrangements of subplots for the best possible readibility for the number of techniques
#you may want to use.
    
    fig,axs=plt.subplots(2,len(VPR_techniques)/2,figsize=(12,6))
    
    row=0
    col=0
    for itr,tech in enumerate(VPR_techniques):
        abc=compute_area_between_the_curves(viewpoint_positions,viewpoint_variations_varied_scores_dict[tech],viewpoint_variations_mismatch_scores_dict[tech])
        viewpoint_variations_ABC_dict[tech]=abc
        
        if (itr<4):
            row=0
            col=itr
        else:
            row=1
            col=itr-4
            
        axs[row,col].plot(range(1,len(viewpoint_positions)+1),viewpoint_variations_varied_scores_dict[tech], label='Same Place')
        axs[row,col].plot(range(1,len(viewpoint_positions)+1),viewpoint_variations_mismatch_scores_dict[tech], label='Different Place')
#        axs[0,itr].set_xticks(viewpoint_positions)
        axs[row,col].set(xlabel='Viewpoint Position', ylabel='Matching Score')
        axs[row,col].title.set_text(tech+', ABC='+str("%0.2f"%abc))
        axs[row,col].legend(loc="upper right")
#        plt.plot(viewpoint_variations_varied_scores_dict[tech], label=tech+' Correct Match')
#        plt.plot(viewpoint_variations_mismatch_scores_dict[tech], label=tech+' False Match')
#        plt.legend()
#        plt.title('Viewpoint Varied, ABC='+str(abc))
#        plt.figure()

    
    fig.tight_layout()
    fig.savefig('Viewpoint_Invariance_Validation__All_VPRBench_PDF_4x2.pdf')  #Saves the figure in the main directory of project.

def perform_vpr_illuminationinvariance_analysis_validation(illumination_quantified_validation_dataset_directory,VPR_techniques):

    illumination_variations_varied_scores_dict={} # Matching scores of same places with viewpoint variations
    illumination_variations_mismatch_scores_dict={} # Matching scores of different places for comparison with above
    illumination_variations_ABC_dict={} #ABC is area-between-the-curves

    illumination_positions=range(25)  #Total 119 different camera positions labelled between 1-119, see paper for more details

    for tech in VPR_techniques:
        matching_scores_varied_with_fixed_viewpoint_and_varied_illumination=np.zeros(len(illumination_positions)) #Image at viewpoint position 1 is matched with itself and with all other different viewpoint positions
        matching_scores_mismatch_with_fixed_viewpoint_and_varied_illumination=np.zeros(len(illumination_positions)) #Image at viewpoint position 1 is matched with other different places 
        
        for illumination in illumination_positions:
            base_illumination=0
            Place1='elm_2floor_bedroom2/'
            Place2='west_kitchen4/'
            base_image_name='dir_'+str(base_illumination)+'_mip2' + '.jpg'    
            base_image=cv2.imread(illumination_quantified_validation_dataset_directory+Place1+base_image_name)       
            varied_image_name='dir_'+str(illumination)+'_mip2' + '.jpg'  
            varied_image=cv2.imread(illumination_quantified_validation_dataset_directory+Place1+varied_image_name)
            mismatch_image_name='dir_'+str(illumination)+'_mip2' + '.jpg'
            mismatch_image=cv2.imread(illumination_quantified_validation_dataset_directory+Place2+mismatch_image_name)
            
            compatible_list_variedimage=[]
            compatible_list_variedimage.append(varied_image)
            
            compatible_list_mismatchimage=[]
            compatible_list_mismatchimage.append(mismatch_image)
       
            if (base_image is not None and varied_image is not None and mismatch_image is not None):
                score_varied,_=match_two_images(base_image,compatible_list_variedimage,tech)
                score_mismatch,_=match_two_images(base_image,compatible_list_mismatchimage,tech)
                matching_scores_varied_with_fixed_viewpoint_and_varied_illumination[illumination]=score_varied
                matching_scores_mismatch_with_fixed_viewpoint_and_varied_illumination[illumination]=score_mismatch
            else:
                print('Images Not Found')                
                
        
        illumination_variations_varied_scores_dict[tech]=matching_scores_varied_with_fixed_viewpoint_and_varied_illumination
        illumination_variations_mismatch_scores_dict[tech]=matching_scores_mismatch_with_fixed_viewpoint_and_varied_illumination


###############################################################################################################################################
#This block of code plots the matching scores for all 8 techniques given illumination variations of MIT Multi-illumination dataset (Fig. 21 of our paper). I have left this code uncommented
#here for your convenience. You would note that it is designed for even number of VPR techniques and works best when all the 8 VPR techniques 
#are being used for analysis. You may want to change the exact arrangements of subplots for the best possible readibility for the number of techniques
#you may want to use.
    
    fig,axs=plt.subplots(2,len(VPR_techniques)/2,figsize=(12,6))
    
    row=0
    col=0
    for itr,tech in enumerate(VPR_techniques):
        abc=compute_area_between_the_curves(illumination_positions,illumination_variations_varied_scores_dict[tech],illumination_variations_mismatch_scores_dict[tech])
        illumination_variations_ABC_dict[tech]=abc
        
        if (itr<4):
            row=0
            col=itr
        else:
            row=1
            col=itr-4
            
        axs[row,col].plot(range(1,len(illumination_positions)+1),illumination_variations_varied_scores_dict[tech], label='Same Place')
        axs[row,col].plot(range(1,len(illumination_positions)+1),illumination_variations_mismatch_scores_dict[tech], label='Different Place')
#        axs[0,itr].set_xticks(viewpoint_positions)
        axs[row,col].set(xlabel='Illumination State', ylabel='Matching Score')
        axs[row,col].title.set_text(tech+', ABC='+str("%0.2f"%abc))
        axs[row,col].legend(loc="upper right")
#        plt.plot(viewpoint_variations_varied_scores_dict[tech], label=tech+' Correct Match')
#        plt.plot(viewpoint_variations_mismatch_scores_dict[tech], label=tech+' False Match')
#        plt.legend()
#        plt.title('Viewpoint Varied, ABC='+str(abc))
#        plt.figure()
    
    fig.tight_layout()
    fig.savefig('Illumination_Invariance_Validation__All_VPRBench_PDF_4x2.pdf')  #Saves the figure in the main directory of project.
        