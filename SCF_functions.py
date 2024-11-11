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
from rasterio.features import geometry_mask
from scipy.spatial import distance


def model_training(curr_acquisition, shapefile_path, SVM_folder_name, gamma=None):
    
    gamma_range = np.logspace(-2, 2, 100)
    
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
    
    if bands_path == []:
        bands_path = glob.glob(os.path.join(curr_acquisition, 'PRS*.tif'))[0]
    else:
        bands_path = bands_path[0]
    
    
    # Load the shapefile
    shapefile = gpd.read_file(shapefile_path)
    
    # Open the bands raster to align with the shapefile
    with rasterio.open(bands_path) as src:
        # Create a mask with the same dimensions as the raster, setting snow (1) and no-snow (2) points
        mask_snow = geometry_mask([geom for geom in shapefile.geometry[shapefile['value'] == 1]],
                                  transform=src.transform,
                                  invert=True,
                                  out_shape=src.shape)
        
        mask_no_snow = geometry_mask([geom for geom in shapefile.geometry[shapefile['value'] == 2]],
                                     transform=src.transform,
                                     invert=True,
                                     out_shape=src.shape)

    # Extract training values using the masks
    snow_training = read_masked_values(bands_path, mask_snow)
    no_snow_training = read_masked_values(bands_path, mask_no_snow)
    
    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    class_array = np.concatenate((np.ones(snow_training.shape[0]), np.zeros(no_snow_training.shape[0])), axis=0)
    
    # Rescale: standardization between 0 and 1
    normalizer = preprocessing.StandardScaler().fit(training_array)
    Samples_train_normalized = normalizer.transform(training_array)
    
    if gamma == None:
        # Gamma selection by examining the kernel
        std_list = []
        for curr_gamma in gamma_range:
            rbf = rbf_kernel(Samples_train_normalized, Samples_train_normalized, gamma=curr_gamma)
            std_list.append(rbf.std())
                               
        idx_max_std = np.argmax(std_list)
        best_gamma = gamma_range[idx_max_std]
        
        print('The best Gamma is: ' + str(best_gamma))
        
    else:
        best_gamma = gamma
        
        
    svm = SVC(C=200000, kernel='rbf', gamma=best_gamma, probability=False,
              decision_function_shape='ovo', cache_size=8000)
    
    svm.fit(Samples_train_normalized, class_array)
    pred = svm.predict(Samples_train_normalized)
   
    svm_model = {'svmModel': svm, 'normalizer': normalizer, 'classes': class_array, 
                 'trainings': training_array, 'SV': svm.support_vectors_}
    
    scf_folder = os.path.join(curr_acquisition, SVM_folder_name)
    if not os.path.exists(scf_folder):
        os.makedirs(scf_folder)
        
    svm_model_filename = os.path.join(scf_folder, 'svm_model.p')
    pickle.dump(svm_model, open(svm_model_filename, "wb")) 
    
    return svm_model_filename


def hyp_disatance(svmModel, svmMatrix):
    return svmModel.decision_function(svmMatrix)


def SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, svm_model_filename, Nprocesses=8, overwrite=False):
    
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0] 
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
    
    if bands_path == []:
        bands_path = glob.glob(os.path.join(curr_acquisition, 'PRS*.tif'))[0]
    else:
        bands_path = bands_path[0]
    scf_folder = os.path.dirname(svm_model_filename)
    
    # Load the SVM model
    svm_model = pickle.load(open(svm_model_filename, 'rb'), encoding='latin1')
    
    with rasterio.open(path_cloud_mask) as src:
        cloud_mask = src.read(1)  # Read the cloud mask (first band)
        
    with rasterio.open(path_water_mask) as src:
        water_mask = src.read(1)  # Read the water mask (first band)
        
    with rasterio.open(bands_path) as src:
        bands = src.read()  
        profile = src.profile
        
    profile.update(dtype='uint8', count=1, compress='lzw', nodata=255, driver='GTiff')
    
    SV = svm_model['SV']
    svm = svm_model['svmModel']
    
    min_score_ns = -1
    max_score_s = 1
    
    valid_mask = np.logical_not(no_data_mask)
    
    FSC_SVM_map_path = os.path.join(scf_folder, os.path.basename(bands_path)[:-11] + '_SnowFLAKES_map.tif')

    # Check if the map file exists and overwrite if specified
    if os.path.exists(FSC_SVM_map_path) and not overwrite:
        print(f"{FSC_SVM_map_path} already exists. Skipping creation.")
        return FSC_SVM_map_path
    
    print('Image classification...\n')
    Image_array_to_classify = bands[:, valid_mask].transpose()
    normalizer = svm_model['normalizer']
    Samples_to_classify = normalizer.transform(Image_array_to_classify)
    
    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)
    
    # Calculate the score
    scoreImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(hyp_disatance)(svm, samplesBlocks[i]) for i in range(len(samplesBlocks))
    )
        
    scoreImage_array = np.concatenate(scoreImage_arrayBlocks, axis=0)
    Score_map = 255 * np.ones(np.shape(valid_mask)).astype(float)
    Score_map[valid_mask] = scoreImage_array.flatten()

    scoreImage_array[scoreImage_array < min_score_ns] = min_score_ns
    scoreImage_array[scoreImage_array > max_score_s] = max_score_s
    
    SCF_Image_array = (scoreImage_array * 50 + 50).astype('uint8')
    
    # Create the SCF map
    SCF_map = 255 * np.ones(np.shape(valid_mask))
    SCF_map[valid_mask] = SCF_Image_array.flatten()
    
    SCF_map[cloud_mask == 2] = 205
    SCF_map[water_mask == 1] = 210
    SCF_map[water_mask == 255] = 210
    
    valid_mask[np.logical_not(valid_mask)] = 255
    
    # Write the SCF map to a file, overwriting if necessary
    with rasterio.open(FSC_SVM_map_path, 'w', **profile) as dst:
        dst.write(SCF_map, 1)
    
    print(f"SCF map saved to {FSC_SVM_map_path}")
    return FSC_SVM_map_path


    
def check_scf_results(FSC_SVM_map_path, shapefile_path, curr_aux_folder, curr_acquisition, k=5, n_closest=5):
    # Define paths for NDSI and bands data
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
    
    if bands_path == []:
        bands_path = glob.glob(os.path.join(curr_acquisition, 'PRS*.tif'))[0]
    else:
        bands_path = bands_path[0]
    
    # Load the SCF map and NDSI map
    with rasterio.open(FSC_SVM_map_path) as scf_src:
        scf_data = scf_src.read(1)  # Reading first band
    
    with rasterio.open(NDSI_path) as ndsi_src:
        ndsi_data = ndsi_src.read(1)  # Reading first band

    # Identify points where SCF > 0 and NDSI < 0
    valid_mask = (scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0)
    
    # Check if there are any valid points; exit if none
    if np.sum(valid_mask) == 0:
        print("No valid points found; exiting function.")
        return shapefile_path  # Optionally return the original shapefile path without modification
    
    print(np.sum(valid_mask))
    
    # Load spectral bands data for these valid points
    with rasterio.open(bands_path) as bands_src:
        bands_data = bands_src.read()  # 3D array (bands, height, width)

    # Use the representative pixel selection function to get new training samples
    representative_pixels_mask = get_representative_pixels(bands_data.reshape(bands_data.shape[0], -1).T, valid_mask.flatten(),
                                                           k=min(k, np.sum(valid_mask.flatten())), n_closest=n_closest).reshape(scf_data.shape)
    
    # Load the original shapefile
    shapefile = gpd.read_file(shapefile_path)
    
    # Convert the representative pixels mask to points in the shapefile's CRS
    new_samples = []
    for row, col in np.argwhere(representative_pixels_mask == 1):
        x, y = scf_src.xy(row, col)  # Get the coordinates
        new_samples.append({'geometry': gpd.points_from_xy([x], [y])[0], 'value': 2})  # 2 for "no snow" label

    # Add new "no snow" samples to the shapefile's GeoDataFrame
    new_samples_gdf = gpd.GeoDataFrame(new_samples, crs=shapefile.crs)
    updated_shapefile = gpd.GeoDataFrame(pd.concat([shapefile, new_samples_gdf], ignore_index=True))
    
    print(f"Additional rows in updated shapefile: {len(updated_shapefile) - len(shapefile)}")
    
    # Save the updated shapefile
    updated_shapefile.to_file(shapefile_path)

    return shapefile_path  





