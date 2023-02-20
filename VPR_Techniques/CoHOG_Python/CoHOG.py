"""
This is the main file for our Python implementation of CoHOG published in IEEE RAL 2020. The HOG_feature folder contains files that have been 
modified by us, however, the original rights for all employed libraries lie with the corresponding contributors and we thanks them for 
open-sourcing their implementations. For the rest of the libraries (skimage, scipy, matplotlib, cv2, numpy etc), please use standard methods of
installation for your IDE, if not already installed.

For any queries or problems, send me an email at mubarizzaffar@gmail.com. If you find this useful, please cite our paper titled 
"CoHOG: A Light-weight, Compute-efficient and Training-free Visual Place Recognition Technique for Changing Environments".

Created on Thu Jan  3 11:18:57 2019

@author: mubariz
"""

import cv2
import numpy as np
from .Hog_feature.Hog_feature.hog import initialize
from .Hog_feature.Hog_feature.hog import extract
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

####################### PARAMETERS #########################
magic_width=512
magic_height=512
cell_size=16  #HOG cell-size
bin_size=8  #HOG bin size
image_frames=1  #1 for grayscale, 3 for RGB
descriptor_depth=bin_size*4*image_frames # x4 is here for block normalization due to nature of HOG
ET=0.4   # Entropy threshold, vary between 0-1.


total_no_of_regions=int((magic_width/cell_size-1)*(magic_width/cell_size-1))

#############################################################

#################### GLOBAL VARIABLES ######################

d1d2dot_matrix=np.zeros([total_no_of_regions,total_no_of_regions],dtype=np.float64)
d1d2matches_maxpooled=np.zeros([total_no_of_regions],dtype=np.float64)
d1d2matches_regionallyweighted=np.zeros([total_no_of_regions],dtype=np.float64)

matched_local_pairs=[]
ref_desc=[]

############################################################


def largest_indices_thresholded(ary):
    good_list=np.where(ary>=ET)
#    no_of_good_regions=len(good_list[0])
    print((len(good_list)))
    print((len(good_list[0])))
    
    return good_list 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
      
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
      

    

#@jit(nopython=False)
def conv_match_dotproduct(d1,d2,regional_gd,total_no_of_regions):            #Assumed aspect 1:1 here
     
    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs
    

    np.dot(d1,d2,out=d1d2dot_matrix)

    np.max(d1d2dot_matrix,axis=1,out=d1d2matches_maxpooled)               #Select best matched ref region for every query region

    np.multiply(d1d2matches_maxpooled,regional_gd,out=d1d2matches_regionallyweighted)   # Weighting regional matches with regional goodness

    score=np.sum(d1d2matches_regionallyweighted)/np.sum(regional_gd)    #compute final match score

    return score
    
##############################################################################################

def compute_query_desc(query):
    
    outlist=[]       
    img_2 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    img_2rgb=query
     
    if (img_2 is not None):
        
        img_2=cv2.resize(img_2,(magic_height,magic_width))
        img_2rgb=cv2.resize(img_2rgb,(magic_height,magic_width))
    
        height,width,angle_unit=initialize(img_2, cell_size, bin_size)
        vector_2 = extract()
        vector_2=np.asfortranarray(vector_2,dtype=np.float32)
        
        ################# Entropy Map ###############################
    #        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_as_ubyte(img_2),(100,100))
    #        ent_time=time.time()
        entropy_image=cv2.resize(entropy(img_gray, disk(5)),(magic_width,magic_height))
    #        print('Entropy Time:',time.time()-ent_time)
        
        ################# Finding Regions #####################
        local_goodness=np.zeros([magic_height/cell_size-1,magic_width/cell_size-1],dtype=np.float32)
        
        for a in range (magic_height/cell_size-1):
            for b in range (magic_width/cell_size-1):
    #                local_staticity=1 #Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a*cell_size:a*cell_size + 2*cell_size, b*cell_size:b*cell_size + 2*cell_size])/(8*(cell_size*4*cell_size))
    
                
                if (local_entropy>=ET):    
                    local_goodness[a,b]=1
                else :
                    #local_goodness[a,b]=local_entropy
                    local_goodness[a,b]=0    

        regional_goodness=local_goodness.flatten()
        print((sum(regional_goodness)))  
        outlist.append(vector_2)
        outlist.append(regional_goodness)
        
        return outlist
        
def compute_map_features(ref_map):  #ref_map is a 1D list of images in this case.

    "INPUT: reference list of images to be matched."
    "OUTPUT: Feature descriptors of all reference images to be matched." 
    ref_desc=[]
    
    for ref in range(len(ref_map)):
            
        img_1 = cv2.cvtColor(ref_map[ref], cv2.COLOR_BGR2GRAY)
    
        if (img_1 is not None):
            
            img_1=cv2.resize(img_1,(magic_height,magic_width))
            
    #        startencodetimer=time.time()
            
            height,width,angle_unit=initialize(img_1, cell_size, bin_size)
            vector_1 = extract()
            
    #        vector_1=np.asarray(vector_1.reshape(total_no_of_regions,len(vector_1[0][0])))
            vector_1=np.asfortranarray(vector_1.transpose(),dtype=np.float32)
      
            ref_desc.append(vector_1)
    print('Reference images descriptors computed!')   
    return ref_desc
    
def perform_VPR(query_info,ref_map_features): #ref_map_features is a 1D list of feature descriptors of reference images in this case.

    "INPUT: Query desc and reference list of images' features to be matched."
    "OUTPUT: Matching Score and Best Matched Image."  


    #        ref_desc[:,:,ref]=vector_1
   
    vector_2=query_info[0]
    regional_goodness=query_info[1]

    confusion_vector=np.zeros(len(ref_map_features),dtype=np.float32)
    ref_desc=ref_map_features
        
    for ref in range(len(ref_map_features)):
        score=conv_match_dotproduct(vector_2.astype('float64'),ref_desc[ref].astype('float64'),regional_goodness,total_no_of_regions)  
        #print(score, ref)
        confusion_vector[ref]=score             
   
            
    return np.amax(confusion_vector), np.argmax(confusion_vector), confusion_vector