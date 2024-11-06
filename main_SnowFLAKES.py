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
   
    #csv_path = os.path.join(current_dir, 'input_csv', 'Generic_area.csv')
    csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/Generic_area.csv')
    input_data = pd.read_csv(csv_path)

    # Retrieve working folder from input data
    working_folder = get_input_param(input_data, 'working_folder')

    # Step 2: Check satellite mission based on acquisition name
    print("Checking satellite mission...")
    acquisitions = os.listdir(working_folder)
    if not acquisitions:
        raise ValueError("No acquisitions found in the working folder.")
    acquisition_name = acquisitions[0]  # Select the first acquisition in the folder
    sensor = get_sensor(acquisition_name)
    
    # Step 3: Create auxiliary folder for permanent layers
    print("Creating auxiliary folder for static data...")
    auxiliary_folder_path = create_auxiliary_folder(working_folder)

    # Step 4: Filter data based on dates and sensor
    start_date = get_input_param(input_data, 'Start Date')
    end_date = get_input_param(input_data, 'End Date')
    print(f"Filtering acquisitions between {start_date} and {end_date}...")
    acquisitions_filtered = data_filter(start_date, end_date, working_folder, sensor)
    
    if not acquisitions_filtered:
        raise ValueError("No acquisitions found for the selected date range and sensor.")

    # Step 5: Retrieve resolution parameter from input data
    resolution = float(get_input_param(input_data, 'resolution'))
    print(f"Using resolution: {resolution}.")

    # Step 6: Create VRT files with stacked bands for the filtered acquisitions
    print("Creating VRT files...")
    create_vrt_files(acquisitions_filtered, sensor, resolution)
    
    print("VRT creation process completed.")
    
    # Step 7: Generate water mask
    print("Generating water mask...")
    ref_img_path=glob.glob(acquisitions_filtered[0]+os.sep+"*scf.vrt")[0]
    vrt,img_info=open_image(ref_img_path)
    # Check if an external water mask is provided
    External_water_mask_path = get_input_param(input_data, 'External_water_mask')
    
    if External_water_mask_path == None:
        water_mask_path = water_identifier(ref_img_path, auxiliary_folder_path)
    else:
        water_mask_path = water_mask_cutting(External_water_mask_path, ref_img_path, auxiliary_folder_path)
    
    print(f"Water mask saved at {water_mask_path}")
    
    # Step 8: Generate glacier mask
    print("Generating glacier mask...")
    
    target_glacier_raster_mask_path = glacier_mask_automatic(water_mask_path)
    
    print(f"Glacier mask saved at {target_glacier_raster_mask_path}")
    
    # Step 9: DEM slope aspect
    
    # Process the image and generate subimage extents
    subimage_extents = process_image(img_info)
    print("Extent computed...")
    
    External_Dem_path = get_input_param(input_data, 'External_water_mask')
    if  External_Dem_path==None:
        print("preparinf DEM download...")
        dem_path = dem_downloader(subimage_extents, ref_img_path, resolution, auxiliary_folder_path, buffer=0.02)
        
        print(f"DEM ready at {dem_path}")
        
    else:
        print("preparinf DEM cropping...")
        dem_path = crop_predefined_DEM(ref_img_path, External_Dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
        print(f"DEM cropped...")
        
    slopePath, aspectPath = calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
    
    scenes_not_to_cloud_mask = []
    
    SVM_folder_name = get_input_param(input_data, 'SVM_folder_name')
    
    for curr_acquisition in acquisitions_filtered:
        
             
            folder=curr_acquisition+os.sep+SVM_folder_name
                    
            if not glob.glob( folder+os.sep+"*FLAKES*")==[] or not glob.glob(  working_folder+os.sep+"*"+SVM_folder_name+"*FLAKES*")==[]:
                print("Scene already processed  "+os.path.basename(working_folder))
                continue 
               
            start = time.time()
           
            print(os.path.basename(curr_acquisition)) 
            
            # Select datetime form  the folder and sensor         
            date_time,date=define_datetime(sensor,curr_acquisition)
            
            # create auxiliary folder for that specific acquisition, it will contain cloud mask, spectral indexes ecc..
            
            curr_aux_folder = create_auxiliary_folder(curr_acquisition, folder_name = '00_auxiliary_folder_' + date)
            
            ## CLOUD MASKING
            
            Compute_clouds = get_input_param(input_data, 'Compute_clouds') == 'yes'
            cloud_prob = float(get_input_param(input_data, 'Cloud cover probability'))
            average_over = int(get_input_param(input_data, 'average_over'))
            dilation_size = int(get_input_param(input_data, 'dilation_cloud_cover'))
            overwrite_cloud = int(get_input_param(input_data, 'Overwrite_cloud'))
            
            path_cloud_mask = os.path.join(curr_aux_folder, os.path.basename(curr_acquisition) + '_cloud_Mask.tif') 

            if not Compute_clouds:
                
                create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                cloud_cover_percentage = 0
            else:
                if sensor == 'S2' and date in scenes_not_to_cloud_mask:
                    create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                    cloud_cover_percentage = 0
                elif sensor == 'S2':
                    
                    stack_clouds_path = [os.path.join(curr_acquisition, f) for f in os.listdir(curr_acquisition) if 'cloud.vrt' in f][0]
                    cloud_mask_path, cloud_cover_percentage = S2_clouds_classifier(stack_clouds_path, path_cloud_mask, ref_img_path, cloud_prob, overwrite_cloud=overwrite_cloud, average_over=average_over, dilation_size=dilation_size)
                
                elif External_cloud_mask_folder:
                    try:
                        path_cloud_mask = glob.glob((curr_acquisition + os.sep + "*cloudMask.tif"))[0]
                    except:
                        path_cloud_mask = glob.glob((External_cloud_mask_folder + os.sep + "*" + os.path.basename(working_folder) + "*.tif"))[0]
                    cloud_mask = open_image(path_cloud_mask)[0]
                    cloud_cover_percentage = np.sum(cloud_mask == 2) / (np.shape(cloud_mask)[0] * np.shape(cloud_mask)[1])
                    del cloud_mask
                    
            ## NO DATA
            
            curr_band_stack_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))[0]
            curr_image, curr_image_info = open_image(curr_band_stack_path)
            no_data_mask, valid_mask = generate_no_data_mask(curr_image, sensor, no_data_value=np.nan)
            
           
            # Calculate no-data percentage
            no_data_percentage = np.sum(no_data_mask) / (curr_image_info['X_Y_raster_size'][0] * curr_image_info['X_Y_raster_size'][1])
            
            # Calculate corrected cloud cover percentage
            clud_perc_corr = cloud_cover_percentage / (1 - no_data_percentage)
            
            
            ## save spectral indexes
            valid_mask = np.logical_not(no_data_mask)
            bands = define_bands(curr_image, valid_mask, sensor)
            
            # NDVI
            spectral_idx_computer(bands['NIR'], bands['RED'], 'NDVI', curr_image, no_data_mask, curr_aux_folder, sensor, sensor + '_' + date + '_NDVI.tif', curr_band_stack_path)
            
            # NDSI
            spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'NDSI', curr_image, no_data_mask, curr_aux_folder, sensor, sensor + '_' + date + '_NDSI.tif', curr_band_stack_path)
            
            # band difference
            spectral_idx_computer(bands['BLUE'], bands['NIR'], 'band_diff', curr_image, no_data_mask, curr_aux_folder, sensor, sensor + '_' + date + '_diffBNIR.tif', curr_band_stack_path)
            
            # Shadow index
            
            spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'shad_idx', curr_image, no_data_mask, curr_aux_folder, sensor, sensor + '_' + date + '_shad_idx.tif', curr_band_stack_path)
        
            # solar incidence angle
            
            solar_incidence_angle = solar_incidence_angle_calculator(curr_image_info, date_time, slopePath, aspectPath, curr_aux_folder, date)
            
            ## Training Collection
            
            result, training_mask_path = collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, bands)
            
            ## training
            
            svm_model_filename = model_training(curr_acquisition, training_mask_path, SVM_folder_name)
            
            ## SCF predict
            
            SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, svm_model_filename, Nprocesses=8)
            
            
            
            

if __name__ == "__main__":
    main()




    