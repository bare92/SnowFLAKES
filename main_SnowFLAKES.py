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

def main():
    # Step 1: Load input data
    # Define the path to the CSV file containing input parameters
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/Azufre.csv')
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/Generic_area.csv')
    #csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/sierra_nevada.csv')
    csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/prisma_test.csv')
    
    input_data = pd.read_csv(csv_path)

    # Retrieve the working folder path from the input CSV file
    working_folder = get_input_param(input_data, 'working_folder')

    # Step 2: Check satellite mission based on acquisition name
    print("Checking satellite mission...")
    acquisitions = sorted(os.listdir(working_folder))
    if not acquisitions:
        raise ValueError("No acquisitions found in the working folder.")
    acquisition_name = acquisitions[-1]  # Use the first acquisition as a representative sample
    sensor = get_sensor(acquisition_name)
    
    # Step 3: Create auxiliary folder for storing permanent layers
    print("Creating auxiliary folder for static data...")
    auxiliary_folder_path = create_auxiliary_folder(working_folder)

    # Step 4: Filter acquisitions by date range and sensor type
    start_date = get_input_param(input_data, 'Start Date')
    end_date = get_input_param(input_data, 'End Date')
    print(f"Filtering acquisitions between {start_date} and {end_date}...")
    acquisitions_filtered = data_filter(start_date, end_date, working_folder, sensor)
    
    if not acquisitions_filtered:
        raise ValueError("No acquisitions found for the selected date range and sensor.")

    # Step 5: Retrieve the spatial resolution parameter
    resolution = float(get_input_param(input_data, 'resolution'))
    print(f"Using resolution: {resolution}.")
    
    # Step 5: Retrieve no data value
    no_data_value = float(get_input_param(input_data, 'no_data_value'))
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
    #target_glacier_raster_mask_path = glacier_mask_automatic(water_mask_path)
    #print(f"Glacier mask saved at {target_glacier_raster_mask_path}")
    
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
        if not Compute_clouds:
            create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
            cloud_cover_percentage = 0
        else:
            if sensor == 'S2' and date in scenes_not_to_cloud_mask:
                create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                cloud_cover_percentage = 0
            elif sensor == 'S2':
                stack_clouds_path = [os.path.join(curr_acquisition, f) for f in os.listdir(curr_acquisition) if 'cloud.vrt' in f][0]
                cloud_mask_path, cloud_cover_percentage = S2_clouds_classifier(
                    stack_clouds_path, path_cloud_mask, ref_img_path, cloud_prob, overwrite_cloud=overwrite_cloud, 
                    average_over=average_over, dilation_size=dilation_size)
            elif External_cloud_mask_folder:
                try:
                    path_cloud_mask = glob.glob((curr_acquisition + os.sep + "*cloudMask.tif"))[0]
                except IndexError:
                    path_cloud_mask = glob.glob((External_cloud_mask_folder + os.sep + "*" + os.path.basename(working_folder) + "*.tif"))[0]
                cloud_mask = open_image(path_cloud_mask)[0]
                cloud_cover_percentage = np.sum(cloud_mask == 2) / (np.shape(cloud_mask)[0] * np.shape(cloud_mask)[1])
                del cloud_mask
            
        # Handle no-data mask
        
        curr_band_stack_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
        
        if curr_band_stack_path == []:
            curr_band_stack_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            curr_band_stack_path = curr_band_stack_path[0]
            
        
        curr_image, curr_image_info = open_image(curr_band_stack_path)
        curr_image[curr_image == no_data_value] = np.nan
        no_data_mask, valid_mask = generate_no_data_mask(curr_image, sensor, no_data_value=np.nan)
        no_data_percentage = np.sum(no_data_mask) / (curr_image_info['X_Y_raster_size'][0] * curr_image_info['X_Y_raster_size'][1])
        cloud_perc_corr = cloud_cover_percentage / (1 - no_data_percentage)
        
        
        if perform_pca:
            
            perform_pca_on_geotiff(curr_band_stack_path, valid_mask)
            
            
        # Compute spectral indices: NDVI, NDSI, band difference, and shadow index
        valid_mask = np.logical_not(no_data_mask)
        bands = define_bands(curr_image, valid_mask, sensor)
        
        spectral_idx_computer(bands['NIR'], bands['RED'], 'NDVI', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NDVI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'NDSI', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_NDSI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['BLUE'], bands['NIR'], 'band_diff', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_diffBNIR.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'shad_idx', curr_image, no_data_mask, curr_aux_folder, sensor, f"{sensor}_{date}_shad_idx.tif", curr_band_stack_path)
        
        # Calculate solar incidence angle
        solar_incidence_angle = solar_incidence_angle_calculator(curr_image_info, date_time, slopePath, aspectPath, curr_aux_folder, date)
        
        # Step 10: Collect training data and train the SVM model
        shapefile_path, training_mask_path = collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path, SVM_folder_name, no_data_mask, bands)
        
        while True:
            
            print('TRAINING')
            svm_model_filename = model_training(curr_acquisition, shapefile_path, SVM_folder_name, gamma = None, perform_pca = perform_pca)
            
            # Step 11: Run SCF prediction
            FSC_SVM_map_path = SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, svm_model_filename, Nprocesses=8, overwrite=True, perform_pca = perform_pca)
            
            # Step 12: Result check
            shapefile_path = check_scf_results(FSC_SVM_map_path, shapefile_path, curr_aux_folder, curr_acquisition, k=5, n_closest=5, perform_pca= perform_pca)
            
            # Load SCF and NDSI data to check the condition
            with rasterio.open(FSC_SVM_map_path) as scf_src:
                scf_data = scf_src.read(1)  # Reading first band
            
            NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
            with rasterio.open(NDSI_path) as ndsi_src:
                ndsi_data = ndsi_src.read(1)  # Reading first band
         
            # Check condition
            if np.sum((scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0)) == 0:
                break  # Exit the loop if no points meet the condition

        print("Process completed. Condition met, and no points found where SCF > 0 and NDSI < 0.")
            
            

if __name__ == "__main__":
    main()




    