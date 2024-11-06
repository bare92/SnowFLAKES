#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:31:29 2024

@author: rbarella
"""

import numpy as np
from training_collection import *
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels, linear_kernel, cosine_similarity
import pickle
from joblib import Parallel, delayed


def model_training(curr_acquisition, training_mask_path, SVM_folder_name):
    
    
    gamma_range = np.logspace(-2, 2, 100)
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))[0]
    
    with rasterio.open(training_mask_path) as src:
        # Read the raster data as a NumPy array
        training_mask = src.read(1)  # '1' refers to the first band
    
    snow_training = read_masked_values(bands_path, training_mask == 1)
    
    no_snow_training = read_masked_values(bands_path, training_mask == 2)
    
    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    
    class_array = np.concatenate((np.zeros(snow_training.shape[0])+1, np.zeros(no_snow_training.shape[0])), axis=0)
    
    # rescale: standardization between 0 and 1
    normalizer = preprocessing.StandardScaler().fit(training_array)
    Samples_train_normalized = normalizer.transform(training_array)
    # gamma selection looking at the kernel
    std_list = []
    for curr_gamma in gamma_range:
                    
        rbf = rbf_kernel(Samples_train_normalized, Samples_train_normalized, gamma = curr_gamma)
        
        std_list.append(rbf.std())
                    
                           
    idx_max_std = np.argmax(std_list)
    best_gamma = gamma_range[idx_max_std]
    
    svm = SVC(C=200000, kernel='rbf', gamma=best_gamma, probability=False,
              decision_function_shape='ovo', cache_size=8000)
    
    svm.fit(Samples_train_normalized, class_array)
    pred = svm.predict(Samples_train_normalized)
    
  
    
    svm_model = {'svmModel': svm, 'normalizer': normalizer, 'classes': class_array, 'trainings': training_array, 'SV': svm.support_vectors_}
    
    scf_folder = os.path.join(curr_acquisition, SVM_folder_name)
    if not os.path.exists(scf_folder):
        os.makedirs(scf_folder)
        
    svm_model_filename = os.path.join(scf_folder, 'svm_model.p')
    pickle.dump(svm_model, open(svm_model_filename, "wb")) 
    
    return svm_model_filename


def hyp_disatance(svmModel, svmMatrix):
    return svmModel.decision_function(svmMatrix)


def SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, svm_model_filename, Nprocesses=8):
    
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0] 
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))[0]
    scf_folder = os.path.dirname(svm_model_filename)
    # Load the svm
    svm_model = pickle.load(open(svm_model_filename, 'rb'), encoding='latin1')
    
    with rasterio.open(path_cloud_mask) as src:
        # Read the raster data as a NumPy array
        cloud_mask = src.read(1)  # '1' refers to the first band
        
    with rasterio.open(path_water_mask) as src:
        # Read the raster data as a NumPy array
        water_mask = src.read(1)  # '1' refers to the first band
        
    with rasterio.open(bands_path) as src:
        # Read the raster data as a NumPy array
        bands = src.read()  
        profile = src.profile
        
    profile.update(dtype='uint8', count=1, compress='lzw', nodata=255, driver='GTiff')
    
    SV = svm_model['SV']
    
    svm = svm_model['svmModel']
    
    
    min_score_ns = -1
    max_score_s = 1
    
    valid_mask = np.logical_not(no_data_mask)

    FSC_SVM_map_path = os.path.join(scf_folder,
                               os.path.basename(bands_path)[:-11] + '_SnowFLAKES_map.tif')


    print('Image classification...\n')
    Image_array_to_classify = bands[:, valid_mask].transpose()
    normalizer = svm_model['normalizer']
    Samples_to_classify = normalizer.transform(Image_array_to_classify)
    # Divide classProb_array in Ncore blocks
    samplesBlocks = np.array_split(Samples_to_classify,Nprocesses,axis=0)
    
    # score
    scoreImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)\
        (delayed(hyp_disatance)(svm, samplesBlocks[i]) for i in range(len(samplesBlocks)))
        
    scoreImage_array = np.concatenate(scoreImage_arrayBlocks, axis=0)
    Score_map = 255 * np.ones(np.shape(valid_mask)).astype(float)
    Score_map[valid_mask] = scoreImage_array.flatten()

    scoreImage_array[scoreImage_array < min_score_ns] = min_score_ns
    scoreImage_array[scoreImage_array > max_score_s] = max_score_s
    
    SCF_Image_array = (scoreImage_array * 50 + 50).astype('uint8')
    
    
    # empty score map

    SCF_map = 255 * np.ones(np.shape(valid_mask))
    
    SCF_map[valid_mask] = SCF_Image_array.flatten()
    
    SCF_map[cloud_mask == 2] = 205
    SCF_map[water_mask == 1] = 210
    SCF_map[water_mask == 255] = 210
    
    valid_mask[np.logical_not(valid_mask)] = 255
    #plt.imshow(SCF_map)
    
    
    if not os.path.exists(FSC_SVM_map_path):
       
        with rasterio.open(FSC_SVM_map_path, 'w', **profile) as dst:
            dst.write(SCF_map, 1)
        
        
    return FSC_SVM_map_path;

    
    




