#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:59:20 2024

@author: rbarella
"""

import os
import pandas as pd
import glob
import logging
from datetime import datetime
from auxiliary_folder_population import *
from utilities import *
import time
import matplotlib.pyplot as plt
from training_collection import *
from SCF_functions import *
from xgboost_functions import *
from shadow_mask_gen import *
from scipy.ndimage import binary_dilation

def main():
    # Step 1: Load input data
    # Define the path to the CSV file containing input parameters
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/azufre.csv')
    csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/maipo_L8.csv')
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/maipo_L7.csv')
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/prisma_test.csv')
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/senales.csv')
    
    input_data = pd.read_csv(csv_path)

    # Retrieve the working folder path from the input CSV file
    working_folder = get_input_param(input_data, 'working_folder')
    
    create_empty_files(working_folder)
    
    scenes_to_skip = scenes_skip(working_folder)
    
    scenes_to_skip_clouds = cloud_mask_to_skip(working_folder)

    # Step 2: Check satellite mission based on acquisition name
    print("Checking satellite mission...")
    acquisitions = sorted(os.listdir(working_folder))
    if not acquisitions:
        raise ValueError("No acquisitions found in the working folder.")
    acquisition_name = acquisitions[-2]  # Use the first acquisition as a representative sample
    sensor = get_sensor(acquisition_name)
    
    # Step 3: Create auxiliary folder for storing permanent layers
    print("Creating auxiliary folder for static data...")
    auxiliary_folder_path = create_auxiliary_folder(working_folder)

    # Step 4: Filter acquisitions by date range and sensor type
    start_date = get_input_param(input_data, 'Start Date')
    end_date = get_input_param(input_data, 'End Date')
    print(f"Filtering acquisitions between {start_date} and {end_date}...")
    acquisitions_filtered = data_filter(start_date, end_date, working_folder, sensor, scenes_to_skip)
    
    if not acquisitions_filtered:
        raise ValueError("No acquisitions found for the selected date range and sensor.")

    # Step 5: Retrieve the spatial resolution parameter
    resolution = float(get_input_param(input_data, 'resolution'))
    print(f"Using resolution: {resolution}.")
    
    # Step 5: Retrieve no data value
    no_data_value = get_input_param(input_data, 'no_data_value')
    
    if no_data_value == None or 'nan' in no_data_value:
        no_data_value = np.nan
        
    else:
        no_data_value = float(no_data_value)
        
    print(f"no data value: {no_data_value}.")

    # Step 6: Generate VRT (Virtual Raster Table) files with stacked bands for selected acquisitions
    print("Creating VRT files...")
    
    if sensor != 'PRISMA':
        create_vrt_files(acquisitions_filtered, sensor, resolution)
        ref_img_path = glob.glob(acquisitions_filtered[0] + os.sep + "*scf.vrt")[0]
        print("VRT creation process completed.")
    if sensor == 'PRISMA':
        
        ref_img_path = [f for f in glob.glob(acquisitions_filtered[0] + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        
    # PCA
    
    if get_input_param(input_data, 'PCA') == 'yes':
        
        perform_pca = True
        
    else:
        perform_pca = False
            
    
    
    # Step 7: Generate water mask
    print("Generating water mask...")
    
    vrt, img_info = open_image(ref_img_path)

    # Check if an external water mask is provided, else generate a default water mask
    External_water_mask_path = get_input_param(input_data, 'External_water_mask')
    if External_water_mask_path is None:
        water_mask_path = water_identifier(ref_img_path, auxiliary_folder_path)
    else:
        water_mask_path = water_mask_cutting(External_water_mask_path, ref_img_path, auxiliary_folder_path)
    
    print(f"Water mask saved at {water_mask_path}")
    
    # Step 8: Generate glacier mask
    print("Generating glacier mask...")
    classify_glaciers = get_input_param(input_data, 'classify_glaciers')
    external_glacier_mask_path = get_input_param(input_data, 'external_glacier_mask_path')
    glaciers_model_svm = get_input_param(input_data, 'glaciers_model_name')
    glaciers_mask_path = glacier_mask_cutting(external_glacier_mask_path, water_mask_path)
    start_glaciers_month = get_input_param(input_data, 'start_glaciers_month')
    end_glaciers_month = get_input_param(input_data, 'end_glaciers_month')
    
    dt_start_glaciers_month = datetime(1900, int(start_glaciers_month), 1)
    dt_end_glaciers_month = datetime(1900, int(end_glaciers_month), 1)
    
    
    
    print(f"Glacier mask saved at {glaciers_mask_path}")
    
    # Step 9: Download or crop DEM, and compute slope and aspect
    subimage_extents = process_image(img_info)
    print("Extent computed...")

    External_Dem_path = get_input_param(input_data, 'External_Dem_path')
    if External_Dem_path is None:
        print("Preparing DEM download...")
        dem_path = dem_downloader(subimage_extents, ref_img_path, resolution, auxiliary_folder_path, buffer=0.02)
    else:
        print("Preparing DEM cropping...")
        dem_path = crop_predefined_DEM(ref_img_path, External_Dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
    
    slopePath, aspectPath = calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
    
    # Initialize an empty list to track scenes without cloud masks
    scenes_not_to_cloud_mask = []
    
    SVM_folder_name = get_input_param(input_data, 'SVM_folder_name')
    
    XGB_folder_name = SVM_folder_name + '_XGB'
    
    # Process each acquisition
    for curr_acquisition in acquisitions_filtered:
        folder = curr_acquisition + os.sep + SVM_folder_name
        
        # Skip already processed scenes
        if glob.glob(folder + os.sep + "*FLAKES*") or glob.glob(working_folder + os.sep + "*" + SVM_folder_name + "*FLAKES*"):
            print("Scene already processed " + os.path.basename(working_folder))
            continue
        
        start = time.time()
        print(os.path.basename(curr_acquisition)) 
        
        # Extract date and time from the folder name and sensor type
        date_time, date = define_datetime(sensor, curr_acquisition)
        
        # Handle no-data mask
        
        curr_band_stack_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
        clouds_band_stack_path = glob.glob(os.path.join(curr_acquisition, '*cloud.vrt'))
        
        if curr_band_stack_path == []:
            curr_band_stack_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            curr_band_stack_path = curr_band_stack_path[0]
            
        if clouds_band_stack_path == []:
            clouds_band_stack_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            clouds_band_stack_path = clouds_band_stack_path[0]
            
        
        curr_image, curr_image_info = open_image(clouds_band_stack_path)
        curr_image[curr_image == no_data_value] = np.nan
        no_data_mask, valid_mask = generate_no_data_mask(curr_image, sensor, no_data_value=np.nan)
        
        # Create auxiliary folder for this acquisition, for storing cloud mask, spectral indices, etc.
        curr_aux_folder = create_auxiliary_folder(curr_acquisition, folder_name='00_auxiliary_folder_' + date)
        
        # Cloud Masking
        Compute_clouds = get_input_param(input_data, 'Compute_clouds') == 'yes'
        cloud_prob = float(get_input_param(input_data, 'Cloud cover probability'))
        average_over = int(get_input_param(input_data, 'average_over'))
        dilation_size = int(get_input_param(input_data, 'dilation_cloud_cover'))
        overwrite_cloud = int(get_input_param(input_data, 'Overwrite_cloud'))
        
        path_cloud_mask = os.path.join(curr_aux_folder, os.path.basename(curr_acquisition) + '_cloud_Mask.tif')
        
        # Generate cloud mask or use default if clouds are not computed
        if not Compute_clouds or date in scenes_to_skip_clouds:
            create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
            cloud_cover_percentage = 0
        else:
            if date in scenes_not_to_cloud_mask:
                create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                cloud_cover_percentage = 0
            elif sensor == 'S2':
                cloud_prob = float(get_input_param(input_data, 'Cloud cover probability'))
                average_over = int(get_input_param(input_data, 'average_over'))
                dilation_size = int(get_input_param(input_data, 'dilation_cloud_cover'))
                overwrite_cloud = int(get_input_param(input_data, 'Overwrite_cloud'))
                stack_clouds_path = [os.path.join(curr_acquisition, f) for f in os.listdir(curr_acquisition) if 'cloud.vrt' in f][0]
                cloud_mask_path, cloud_cover_percentage = S2_clouds_classifier(
                    stack_clouds_path, path_cloud_mask, ref_img_path, cloud_prob, overwrite_cloud=overwrite_cloud, 
                    average_over=average_over, dilation_size=dilation_size)
            elif sensor == 'L7' or sensor == 'L8':
               
                path_cloud_mask, cloud_cover_percentage = landsat_cloud_classifier(curr_aux_folder, path_cloud_mask, ref_img_path, sensor, valid_mask, Nprocesses=8, dilate_iterations=5)
            


        no_data_percentage = np.sum(no_data_mask) / (curr_image_info['X_Y_raster_size'][0] * curr_image_info['X_Y_raster_size'][1])
        cloud_perc_corr = cloud_cover_percentage / (1 - no_data_percentage)
        
        
        if perform_pca:
            
            perform_pca_on_geotiff(curr_band_stack_path, valid_mask)
            
            
        # Compute spectral indices: NDVI, NDSI, band difference, and shadow index
        valid_mask = np.logical_not(no_data_mask)
        
        if np.sum(no_data_mask) /len(valid_mask.flatten()) > 1 or cloud_perc_corr > 0.5:
            
            print('TOO MANY INVALID PIXELS...')
            
            continue
            
        bands = define_bands(curr_image, valid_mask, sensor)
        
        spectral_idx_computer(bands['NIR'], bands['RED'], 'normDiff', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NDVI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'normDiff', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NDSI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['BLUE'], bands['NIR'], 'band_diff', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_diffBNIR.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'shad_idx', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_shad_idx.tif", curr_band_stack_path)
        
        spectral_idx_computer(bands['BLUE'], bands['NIR'], 'normDiff', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NormDiffBNIR.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['RED'], 'normDiff', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NormDiffGreenRed.tif", curr_band_stack_path)
        
        spectral_idx_computer(bands['NIR'], bands['RED'], 'EVI', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_EVI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['RED'], 'NDSIplus', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NDSIplus.tif", curr_band_stack_path, B3=bands['NIR'], B4=bands['SWIR'])
        spectral_idx_computer(bands['GREEN'], bands['RED'], 'idx6', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_idx6.tif", curr_band_stack_path, B3=bands['NIR'])
        
        # Calculate solar incidence angle
        solar_incidence_angle, sun_altitude, sun_azimuth = solar_incidence_angle_calculator(curr_image_info, date_time, slopePath, aspectPath, curr_aux_folder, date)
        
        # shadow mask
        
        shadow_mask_path = generate_shadow_mask(curr_aux_folder, auxiliary_folder_path, no_data_mask, bands['NIR'])
        
        
        # Step 10: Collect training data and train the SVM model
        
        if classify_glaciers == 'yes' and date_time.month >= dt_start_glaciers_month.month and date_time.month <= dt_end_glaciers_month.month: 
            with rasterio.open(glaciers_mask_path) as src:
                glaciers_mask = src.read(1)  # Read the cloud mask (first band)
                
                # Apply an N-pixel buffer using binary dilation
            N = 3  # Replace this with the desired buffer size
            structure = np.ones((2 * N + 1, 2 * N + 1))  # Define the dilation kernel
            glaciers_mask = binary_dilation(glaciers_mask, structure=structure).astype(int)

            training_collection_no_data_mask = np.logical_or(no_data_mask, glaciers_mask == 1)
            
        else:
            training_collection_no_data_mask = no_data_mask
        
        shapefile_path, training_mask_path = collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path, SVM_folder_name, training_collection_no_data_mask, bands, shadow_mask_path)
        
        ## Preclassification with xgboost
        
        NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
        with rasterio.open(NDSI_path) as ndsi_src:
            ndsi_data = ndsi_src.read(1)  # Reading first band
        counter_to_exit = 0
        while True:
            
            print('TRAINING')
            svm_model_filename = model_training(curr_acquisition, shapefile_path, SVM_folder_name, gamma = None, perform_pca = perform_pca)
            
            #xgb_model_filename = model_training_xgb(curr_acquisition, shapefile_path, XGB_folder_name, perform_pca=False, grid_search=True)
            
            # Step 11: Run SCF prediction
            FSC_SVM_map_path = SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, svm_model_filename, Nprocesses=1, overwrite=True, perform_pca = perform_pca)
            # FSC_XGB_map_path = snow_class_XGB(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, xgb_model_filename, 
            #                  Nprocesses=8, overwrite=true, perform_pca=False)
            
            # Step 12: Result check
            shapefile_path = check_scf_results(FSC_SVM_map_path, shapefile_path, curr_aux_folder, curr_acquisition, k=5, n_closest=5, perform_pca= perform_pca)
            
            # Load SCF and NDSI data to check the condition
            with rasterio.open(FSC_SVM_map_path) as scf_src:
                scf_data = scf_src.read(1)  # Reading first band
                
            # Load SM and NDSI data to check the condition
            # with rasterio.open(FSC_XGB_map_path) as sm_src:
            #     sm_data = sm_src.read(1)  # Reading first band
            
            
         
            # Check condition
            #if np.sum((scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0) & (sm_data <= 100) & (sm_data > 0)) == 0 or counter_to_exit >= 10:
                
            if np.sum((scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0)) == 0 or counter_to_exit >= 10:
                break  # Exit the loop if no points meet the condition
                
            counter_to_exit += 1
            
        ## Glacier_classification
        emisphere = get_hemisphere(FSC_SVM_map_path)
        
        # if classify_glaciers == 'yes' and date_time.month >= dt_start_glaciers_month.month and date_time.month <= dt_end_glaciers_month.month:
        #     glaciers_classifier(FSC_SVM_map_path, auxiliary_folder_path, glaciers_model_svm, curr_acquisition, Nprocesses=1)

        print("Process completed. Condition met, and no points found where SCF > 0 and NDSI < 0.")
            
        give_time(start) 

if __name__ == "__main__":
    main()




    