#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:16:55 2024

@author: rbarella
"""

import os
import glob
from osgeo import gdal
import numpy as np

def data_filter(start_date, end_date, working_folder, sensor):
    """
    Filters the scenes in the working folder by date and sensor.

    Parameters
    ----------
    start_date : str
        Start date in the format "yyyymmdd".
    end_date : str
        End date in the format "yyyymmdd".
    working_folder : str
        The folder containing the acquisitions.
    sensor : str
        The sensor identifier (e.g., "S2", "L8").

    Returns
    -------
    list
        A list of filtered scene directories between the provided dates.
    """
    # Filter for Sentinel-2
    if sensor == 'S2':
        acquisitions = sorted(glob.glob(os.path.join(working_folder, 'S2*')))
        acquisitions_filtered = [
            f for f in acquisitions
            if (
                # MSIL1C format
                (os.path.basename(f).split('_')[1] == 'MSIL1C' and 
                 start_date <= os.path.basename(f).split('_')[2].split('T')[0] <= end_date) or
                # OPER format
                (os.path.basename(f).split('_')[1] == 'OPER' and 
                 start_date <= os.path.basename(f).split('_')[7][1:].split('T')[0] <= end_date)
            )
        ]
    
    # Filter for Landsat (L8, L7, etc.)
    else:
        acquisitions = sorted(glob.glob(os.path.join(working_folder, 'L*T1')))
        acquisitions_filtered = [
            f for f in acquisitions
            if start_date <= os.path.basename(f).split('_')[3] <= end_date
        ]
    
    print(f"Filtered acquisitions...\n")
    return acquisitions_filtered

def get_input_param(input_data, name):
    """Retrieves the value associated with a given name from a DataFrame."""
    filtered_data = input_data[input_data['Name'] == name]
    value = input_data[input_data['Name'] == 'External_water_mask']['Value'].values[0]
    if np.isnan(value):
        return None
    if len(filtered_data) > 1:
        raise ValueError(f"Multiple matching rows found for name: {name}")
    return filtered_data['Value'].iloc[0]

def get_sensor(acquisition_name):
    """Determines the satellite mission based on the acquisition name."""
    acquisition_name = os.path.basename(acquisition_name)
    if 'LT04' in acquisition_name:
        return 'L4'
    elif 'LT05' in acquisition_name or acquisition_name[:3] == 'LT5':
        return 'L5'
    elif 'LE07' in acquisition_name or acquisition_name[:3] == 'LE7':
        return 'L7'
    elif 'LC08' in acquisition_name or acquisition_name[:3] == 'LC8':
        return 'L8'
    elif 'LC09' in acquisition_name:
        return 'L8'
    elif 'S2' in acquisition_name:
        return 'S2'
    else:
        raise ValueError(f"Invalid acquisition name: {acquisition_name}")

def create_vrt_files(L_acquisitions_filt, sensor, resolution):
    """
    Creates VRT files for each acquisition based on the sensor type.
    """
    for elem in L_acquisitions_filt:
        if sensor == 'S2':
            create_vrt(elem, elem, 'cloud', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'scf', resolution=resolution, overwrite=False)
        else:
            create_vrt(elem, elem, 'scf', resolution=resolution, overwrite=False)

def create_vrt(folder, outfolder, suffix="scf", resolution=30, overwrite=False):
    """
    Creates a VRT using selected bands for the specified sensor.
    """
    vrtname = os.path.join(outfolder, os.path.basename(folder) + f'_{suffix}.vrt')
    
    if os.path.exists(vrtname) and not overwrite:
        print(f"{vrtname} already exists.")
        return

    # Determine the sensor type and get the bands
    sensor = get_sensor(folder)
    band_name_list = select_band_names(sensor, suffix)
    
    # Find the band files in the folder
    file_list = find_band_files(folder, band_name_list, sensor)
    if not file_list:
        print(f"No band files found for sensor: {sensor}, suffix: {suffix}.")
        return

    # Create the VRT using GDAL
    create_vrt_with_gdal(file_list, vrtname, resolution, band_name_list)

def select_band_names(sensor, suffix):
    """
    Returns the list of band names based on the sensor and suffix.
    """
    if sensor in ['L4', 'L5', 'L7']:
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    elif sensor == 'L8':
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    elif sensor == 'S2' and suffix == 'cloud':
        return ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    elif sensor == 'S2' and suffix == 'scf':
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

def find_band_files(folder, band_name_list, sensor):
    """
    Finds the file paths for the selected bands in the folder.
    """
    file_list = []
    for band in band_name_list:
        pattern = os.path.join(folder, f"*{band}_*.tif")
        file_list.extend(glob.glob(pattern))

    # Handle Landsat (non-S2) cases with uppercase .TIF extension
    if not file_list and sensor != 'S2':
        for band in band_name_list:
            pattern = os.path.join(folder, f"*{band}.TIF")
            file_list.extend(glob.glob(pattern))

    # Handle Sentinel-2 .tif extension cases
    if sensor == 'S2' and not file_list:
        for band in band_name_list:
            pattern = os.path.join(folder, f"*{band}.tif")
            file_list.extend(glob.glob(pattern))

    return file_list

def create_vrt_with_gdal(file_list, vrtname, resolution, band_name_list):
    """
    Creates a VRT file using GDAL.
    """
    file_string = " ".join(file_list)
    cmd = f"gdalbuildvrt -separate -r bilinear -tr {resolution} {resolution} {vrtname} {file_string}"
    print(f"Running command: {cmd}")
    os.system(cmd)

    # Set the band descriptions in the VRT
    VRT_dataset = gdal.Open(vrtname, gdal.GA_Update)
    for idx, band_name in enumerate(band_name_list, 1):
        VRT_dataset.GetRasterBand(idx).SetDescription(band_name)
    VRT_dataset = None
























