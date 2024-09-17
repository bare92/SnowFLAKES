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

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Step 1: Load input data
   
    #csv_path = os.path.join(current_dir, 'input_csv', 'Generic_area.csv')
    csv_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/SnowFLAKES/input_csv/Generic_area.csv')
    input_data = pd.read_csv(csv_path)

    # Retrieve working folder from input data
    working_folder = get_input_param(input_data, 'working_folder')

    # Step 2: Check satellite mission based on acquisition name
    log.info("Checking satellite mission...")
    acquisitions = os.listdir(working_folder)
    if not acquisitions:
        raise ValueError("No acquisitions found in the working folder.")
    acquisition_name = acquisitions[0]  # Select the first acquisition in the folder
    sensor = get_sensor(acquisition_name)
    
    # Step 3: Create auxiliary folder for permanent layers
    log.info("Creating auxiliary folder for static data...")
    auxiliary_folder_path = create_auxiliary_folder(working_folder)

    # Step 4: Filter data based on dates and sensor
    start_date = get_input_param(input_data, 'Start Date')
    end_date = get_input_param(input_data, 'End Date')
    log.info(f"Filtering acquisitions between {start_date} and {end_date}...")
    acquisitions_filtered = data_filter(start_date, end_date, working_folder, sensor)
    
    if not acquisitions_filtered:
        raise ValueError("No acquisitions found for the selected date range and sensor.")

    # Step 5: Retrieve resolution parameter from input data
    resolution = float(get_input_param(input_data, 'resolution'))
    log.info(f"Using resolution: {resolution}.")

    # Step 6: Create VRT files with stacked bands for the filtered acquisitions
    log.info("Creating VRT files...")
    create_vrt_files(acquisitions_filtered, sensor, resolution)
    
    log.info("VRT creation process completed.")
    
    # Step 7: Generate water mask
    log.info("Generating water mask...")
    ref_img_path=glob.glob(acquisitions_filtered[0]+os.sep+"*scf.vrt")[0]
    vrt,img_info=open_image(ref_img_path)
    # Check if an external water mask is provided
    External_water_mask_path = get_input_param(input_data, 'External_water_mask')
    
    if External_water_mask_path == None:
        water_mask_path = water_identifier(ref_img_path, auxiliary_folder_path)
    else:
        water_mask_path = water_mask_cutting(External_water_mask_path, ref_img_path, auxiliary_folder_path)
    
    log.info(f"Water mask saved at {water_mask_path}")
    
    # Step 8: Generate glacier mask
    log.info("Generating glacier mask...")
    
    target_glacier_raster_mask_path = glacier_mask_automatic(water_mask_path)
    
    log.info(f"Glacier mask saved at {target_glacier_raster_mask_path}")
    
    # Step 9: DEM slope aspect
    log.info("preparinf subimages extent...")
    # Process the image and generate subimage extents
    subimage_extents = process_image(img_info)
    log.info("Extent computed...")
    
    External_Dem_path = get_input_param(input_data, 'External_water_mask')
    if  External_Dem_path==None:
        log.info("preparinf DEM download...")
        dem_path = dem_downloader(subimage_extents, ref_img_path, resolution, auxiliary_folder_path, buffer=0.02)
        
        log.info(f"DEM ready at {dem_path}")
        
    else:
        log.info("preparinf DEM cropping...")
        dem_path = crop_predefined_DEM(ref_img_path, External_Dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
        log.info(f"DEM cropped...")
        
    slopePath, aspectPath = calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
    
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
                       

            if not Compute_clouds:
                path_cloud_mask = curr_acquisition + os.sep + os.path.basename(curr_acquisition) + '_cloud_Mask.tif'
                create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                cloud_cover_percentage = 0
            else:
                if sensor == 'S2' and date in scenes_not_to_cloud_mask:
                    create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                    cloud_cover_percentage = 0
                elif sensor == 'S2':
                    stack_clouds_path = [os.path.join(curr_acquisition, f) for f in os.listdir(curr_acquisition) if 'cloud.vrt' in f][0]
                    path_cloud_mask, cloud_cover_percentage = S2_clouds_classifier(stack_clouds_path, stack_fsc_path, cloud_prob, Overwrite_cloud, average_over=average_over, dilation_size=average_over)
                elif External_cloud_mask_folder:
                    try:
                        path_cloud_mask = glob.glob((curr_acquisition + os.sep + "*cloudMask.tif"))[0]
                    except:
                        path_cloud_mask = glob.glob((External_cloud_mask_folder + os.sep + "*" + os.path.basename(working_folder) + "*.tif"))[0]
                    cloud_mask = open_image(path_cloud_mask)[0]
                    cloud_cover_percentage = np.sum(cloud_mask == 2) / (np.shape(cloud_mask)[0] * np.shape(cloud_mask)[1])
                    del cloud_mask
            

if __name__ == "__main__":
    main()




    