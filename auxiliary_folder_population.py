#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:39:26 2024

@author: rbarella
"""

import os
import logging
from osgeo import gdal, osr
import geopandas as gpd
import numpy as np
import cv2
from shapely.geometry import Point,Polygon 
from scipy.ndimage import median_filter
import elevation
import glob
import pandas as pd
from utilities import *
import rasterio
from pyproj import Transformer
from datetime import timezone
from pysolar.solar import *
from rasterio.crs import CRS
from pyproj import Transformer
from pathlib import Path

def create_auxiliary_folder(working_folder, folder_name = '01_TEST_auxiliary_folder'):
    """
    Creates an auxiliary folder in the working directory for storing permanent layers (e.g., DEM, masks).
    
    Parameters
    ----------
    working_folder : str
        The main working directory where the auxiliary folder will be created.

    Returns
    -------
    str
        The path of the auxiliary folder.
    """
    # Define path for the ancillary folder
    auxiliary_folder_path = os.path.join(working_folder, folder_name)

    # Check if the folder exists, create if not
    if not os.path.exists(auxiliary_folder_path):
        os.makedirs(auxiliary_folder_path)
        logging.info(f"Auxiliary folder created at {auxiliary_folder_path}.")
    else:
        logging.info(f"Auxiliary folder already exists at {auxiliary_folder_path}.")

    return auxiliary_folder_path

def water_identifier(ref_img_path,auxiliary_folder_path):
    '''
    This function cut the water mask from the copernicus on the extent given by a ref image
    https://global-surface-water.appspot.com/download
    

    Parameters
    ----------
    ref_img_path : str
        path from a refernce image to cut water mask on .
    ancillary : bool, optional
        Presnce of ancillry folder, to store or not there the water mask. The default is False.

    Returns
    -------
    target_wb_mask_path : str
        water mask path.

    '''
    
   
    target_wb_mask_path = auxiliary_folder_path +os.sep+os.path.basename(os.path.dirname( os.path.dirname(ref_img_path)))+ "_Water_Mask.tif"  
   
        
        
    if not os.path.exists(target_wb_mask_path):
        dim_img=open_image(ref_img_path)[1]
        
        with rasterio.open(ref_img_path, 'r+') as rds:
            epsg_code = str(rds.crs).split(':')[1]
        
        resolution=dim_img['geotransform'][1]
        E_min=dim_img["extent"][0]
        N_min=dim_img["extent"][1]
        E_max=dim_img["extent"][2]
        N_max=dim_img["extent"][3]
        
      
        if epsg_code != "4326":
            # Open the reference image to get its CRS and extent
            with rasterio.open(ref_img_path) as src:
                srIn = src.crs  # Reference image CRS
                E_min_old, N_min_old, E_max_old, N_max_old = src.bounds  # Original bounds in source CRS
                resolution = src.res[0]  # Assumes square pixels for resolution
        
            # Get the target CRS from the water mask file
            water_mask_file = glob.glob("/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/*")[0]
            with rasterio.open(water_mask_file) as d_target:
                srOut = d_target.crs  # Water mask CRS
        
            # Set up the transformation between the two coordinate reference systems (CRS)
            transformer = Transformer.from_crs(srIn, srOut, always_xy=True)
        
            # Reproject the bounds from the source CRS (reference image) to the target CRS (water mask)
            E_min, N_min = transformer.transform(E_min_old, N_min_old)
            E_max, N_max = transformer.transform(E_max_old, N_max_old)
        
            # Adjust the resolution scale if needed
            resolution /= 100000
                    
        V1=(int(np.floor(E_min/10)*10),int(np.ceil(N_min/10)*10))
        V2=(int(np.floor(E_min/10)*10),int(np.ceil(N_max/10)*10))
        V3=(int(np.floor(E_max/10)*10),int(np.ceil(N_min/10)*10))
        V4=(int(np.floor(E_max/10)*10),int(np.ceil(N_max/10)*10))
        
        V_LIST=[V1,V2,V3,V4]
        nome_tile=[]
        for v in V_LIST:
            if v[0]>=0 :
                E=str(int(np.floor( v[0]/10)*10))
                W=None
                lat="E"
            else:
                W=str(int(abs(np.floor( v[0]/10) *10 )))                
                E=None
                lat="W"
            if v[1]>=0 :
                N=str(int(np.ceil( v[1]/10)*10))                
                S=None
                lon="N"
            else:
                S=str(int(abs(np.floor( v[1]/10) *10 )))                
                N=None
                lon="S"
           
            if W==None and N==None:
                nome="extent_"+E+lat+"_"+S+lon+"v1_4_2021.tif"
            if W==None and S==None:
                nome="extent_"+E+lat+"_"+N+lon+"v1_4_2021.tif"
            if E==None and N==None:
                nome="extent_"+W+lat+"_"+S+lon+"v1_4_2021.tif"
            if E==None and S==None:
                nome="extent_"+W+lat+"_"+N+lon+"v1_4_2021.tif"
            file="/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/"+nome
            if not file in     nome_tile:
                
                nome_tile.append(file)
        
        filenames = ' '.join(nome_tile)
            
        extent_string = ' '.join([str(E_min), str(N_min), str(E_max), str(N_max)])
                   
        if  epsg_code!="4326":
            
            temp_file=target_wb_mask_path[:-4]+"_temp.tif"
            
            cmd = 'gdalbuildvrt -r bilinear -te ' + extent_string+" -tr " + str(resolution) + ' ' + str(resolution) + ' ' + temp_file + ' ' + filenames               
            os.system(cmd)
          
            cmd = 'gdalwarp -t_srs EPSG:'+epsg_code+' -te ' + ' '.join((str(E_min_old), str(N_min_old), str(E_max_old), str(N_max_old))) +\
                ' -r bilinear -tr ' + ' '.join((str(resolution*100000), str(resolution*100000)))+ ' '  + ' '.join((temp_file,target_wb_mask_path ))            
            os.system(cmd)
            os.remove(temp_file)                      
        else:    
            cmd = 'gdalbuildvrt -r bilinear -te ' + extent_string+" -tr " + str(resolution) + ' ' + str(resolution) + ' ' + target_wb_mask_path + ' ' + filenames   
            os.system(cmd)
            
        water_mask=open_image(target_wb_mask_path)[0]    
        if   np.sum(water_mask==255) >0: 
            
            K= np.ones((30,30)).astype(np.uint8)  
            Water_dilated = cv2.dilate((water_mask==255).astype(np.uint8), K, iterations=1) 
            # create a single water mask with 0-1 
            water_mask[Water_dilated==1]=255 
            water_mask[water_mask==210]=1
            water_mask[water_mask==255]=1   
            os.remove(target_wb_mask_path)
            save_image(water_mask.astype('uint8'), target_wb_mask_path, 'GTiff', 1, dim_img['geotransform'], dim_img['projection'])
                    
    return target_wb_mask_path

def water_mask_cutting(water_mask_path, ref_img_path,auxiliary_folder_path):
    '''
    

    Parameters
    ----------
    water_mask_path : str
        path of water mask to cut .
    ref_img_path : str
        path of a reference image.
    Ancillary_folder : bool


    Returns
    -------
    target_wb_mask_path : str
        water mask path.


    '''
    if auxiliary_folder_path != None:
        target_wb_mask_path = auxiliary_folder_path+os.sep+os.path.basename(os.path.dirname( os.path.dirname(ref_img_path)))+ "_Water_Mask.tif"  
    else:
        target_wb_mask_path = ref_img_path[:-8] + "_Water_Mask.tif"  
    
    if not os.path.exists(target_wb_mask_path):
    # clip the wbm with FSC extent
      
        img_info = open_image(ref_img_path)[1]
    
        d = gdal.Open(ref_img_path)
        with rasterio.open(ref_img_path, 'r+') as rds:
            epsg_code_ref = str(rds.crs).split(':')[1]
            
        E_min = (img_info['extent'][0])
        N_min = (img_info['extent'][1])
        E_max = (img_info['extent'][2])
        N_max = (img_info['extent'][3])
        img_res = str(img_info['geotransform'][1])
        
        extent_string = ' '.join([str(E_min), str(N_min), str(E_max), str(N_max)])
        cmd = 'gdalwarp -t_srs EPSG:' + epsg_code_ref + ' -te ' + extent_string + ' -tr ' + ' '.join([img_res, img_res]) + \
            ' -of GTiff ' + ' '.join([water_mask_path,target_wb_mask_path])
        
        os.system(cmd)
       
        water_mask=open_image(target_wb_mask_path)[0]
       
        #dialte the nan value (255) of the water mask into the 1 value of water mask 
        if   np.sum(water_mask==255) >0: 
            K= np.ones((30,30)).astype(np.uint8)  
            Water_dilated = cv2.dilate((water_mask==255).astype(np.uint8), K, iterations=1) 
            # create a single water mask with 0-1 
            water_mask[Water_dilated==1]=255 
            water_mask[water_mask==210]=1
            water_mask[water_mask==255]=1   
            os.remove(target_wb_mask_path)
            save_image(water_mask.astype('uint8'), target_wb_mask_path, 'GTiff', 1, img_info['geotransform'], img_info['projection'])
                    
    return target_wb_mask_path;

def glacier_mask_automatic(water_mask_path):
    """
    Automatically selects the glacier mask from the global NASA inventory taken from 
    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v6/. 
    If the area does not contain glaciers, it returns a raster layer filled with zeros.

    Parameters
    ----------
    water_mask_path : str
        Path to the water mask.
    ancillary_folder : bool, optional
        If True, the glacier mask will be stored in the ancillary folder (default is False).
    
    Returns
    -------
    str
        Path to the .tif file containing the glacier mask.
    """
    
    # Define target paths for glacier shapefile and raster
    base_path = water_mask_path[:-14]
    target_glacier_shp_mask_path = base_path + 'glacier.shp'
    target_glacier_raster_mask_path = base_path + 'glacier.tif'
    
    # If glacier raster mask already exists, return its path
    if os.path.exists(target_glacier_raster_mask_path):
        return target_glacier_raster_mask_path
    
    # Define path to glacier inventory data
    glacier_inventory_folder = "/mnt/CEPH_PROJECTS/ALPSNOW/Nicola/Glacer_mask/Glacier_inventory"
    shapefiles = glob.glob(glacier_inventory_folder + os.sep + "*/*shp")
    
    # Initialize lists to store extent information of glacier shapefiles
    extents = {'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'filename': []}
    polygons = []
    
    # Loop over glacier shapefiles to store extents and filenames
    for shape in shapefiles:
        shapefile = ogr.Open(shape)
        layer = shapefile.GetLayer(0)
        extent = layer.GetExtent()
        
        # Store extent information
        extents['xmin'].append(extent[0])
        extents['ymin'].append(extent[2])
        extents['xmax'].append(extent[1])
        extents['ymax'].append(extent[3])
        extents['filename'].append(shape)
        
        # Create and store corresponding polygons
        polygons.append(Polygon([(extent[0], extent[2]), (extent[1], extent[2]), 
                                 (extent[1], extent[3]), (extent[0], extent[3])]))
    
    # Create a GeoDataFrame for extent information
    extent_df = pd.DataFrame(extents)
    gdf = gpd.GeoDataFrame(extent_df, crs="EPSG:4326", geometry=polygons)
    
    # Open the water mask image and get its extent and projection
    img, img_info = open_image(water_mask_path)
    d = gdal.Open(water_mask_path)
    proj = osr.SpatialReference(wkt=d.GetProjection())
    epsg_code = proj.GetAttrValue("AUTHORITY", 1)
    resolution = img_info['geotransform'][1]
    img_res = str(resolution)
    
    # Extract image extent based on EPSG code (whether WGS 84 or another CRS)
    if epsg_code == "4326":
        E_min, N_min, E_max, N_max = img_info['extent']
    else:
        srIn = osr.SpatialReference(str(img_info['projection']))
       
        srOut = osr.SpatialReference(str(open_image(water_mask_path)[1]['projection']))
        
        E_min_meter, N_min_meter, E_max_meter, N_max_meter = img_info['extent']
        N_min, E_min = reproj_point(E_min_meter, N_min_meter, srIn, srOut)
        N_max, E_max = reproj_point(E_max_meter, N_max_meter, srIn, srOut)
        resolution /= 100000  # Adjust resolution for meters-to-degrees conversion
        img_res = str(resolution)
    
    # Calculate the geometric center of the extent
    E_mean = (E_min + E_max) / 2
    N_mean = (N_min + N_max) / 2
    center_point = Point(E_mean, N_mean)
    
    # Check if any glacier polygon contains the center of the image extent
    if gdf['geometry'].contains(center_point).any():
        glacier_outlines_shp_path = gdf['filename'][gdf['geometry'].contains(center_point)].values[0]
    else:
        # Fallback glacier shapefile for areas with no glaciers
        glacier_outlines_shp_path = "/mnt/CEPH_PROJECTS/ALPSNOW/Nicola/Glacer_mask/Alps_glacier/c3s_gi_rgi11_s2_2015_v2.shp"
    
    # Create a polygon corresponding to the image extent
    extent_polygon = Polygon([(E_min, N_min), (E_min, N_max), (E_max, N_max), (E_max, N_min)])
    poly_gdf = gpd.GeoDataFrame([1], geometry=[extent_polygon], crs="EPSG:4326")
    
    # Read and clip the glacier outlines shapefile
    glacier_outlines_shp = gpd.read_file(glacier_outlines_shp_path)
    
    if epsg_code == "4326":
        glacier_outlines_shp_reproj = glacier_outlines_shp.to_crs("EPSG:4326")
        glacier_outlines_shp_reproj['geometry'] = glacier_outlines_shp_reproj.buffer(0)  # Fix any geometry issues
        clipped_glaciers = gpd.clip(glacier_outlines_shp_reproj, poly_gdf)
    else:
        clipped_glaciers = gpd.clip(glacier_outlines_shp, poly_gdf)
    
    # If the clipped glaciers are not empty, rasterize the glacier outlines
    if not clipped_glaciers.empty:
        clipped_glaciers.to_file(target_glacier_shp_mask_path)
        
        if epsg_code == "4326":
            # Rasterize glacier outlines
            rasterize_cmd = f"gdal_rasterize -burn 1 -a_nodata 0 -te {E_min} {N_min} {E_max} {N_max} -tr {img_res} {img_res} {target_glacier_shp_mask_path} {target_glacier_raster_mask_path}"
            os.system(rasterize_cmd)
        else:
            temp_file = target_glacier_raster_mask_path.replace(".tif", "_temp.tif")
            rasterize_cmd = f"gdal_rasterize -burn 1 -a_nodata 0 -te {E_min} {N_min} {E_max} {N_max} -tr {img_res} {img_res} {target_glacier_shp_mask_path} {temp_file}"
            os.system(rasterize_cmd)
            
            # Reproject the raster to the original CRS
            warp_cmd = f"gdalwarp -s_srs EPSG:4326 -t_srs EPSG:{epsg_code} -te {E_min_meter} {N_min_meter} {E_max_meter} {N_max_meter} -r near -tr {resolution*100000} {resolution*100000} {temp_file} {target_glacier_raster_mask_path}"
            os.system(warp_cmd)
            
            os.remove(temp_file)
            os.remove(target_glacier_shp_mask_path)
    else:
        # If no glaciers are found, create a raster filled with zeros
        glacier_mask = np.zeros_like(img)
        save_image(glacier_mask, target_glacier_raster_mask_path, 'GTiff', 1, img_info['geotransform'], img_info['projection'])
    
    return target_glacier_raster_mask_path

def dem_downloader(sub_areas, ref_img_path, resolution, auxiliary_folder_path, buffer=0.02):
    """
    Downloads and mosaics a DEM (Digital Elevation Model) for a given set of sub-areas. If a DEM
    already exists, the function skips downloading. If not, it downloads the DEM, reprojects, and
    mosaics the sub-areas into one file.
    Parameters
    ----------
    sub_areas : list of tuples
        List of areas to download DEMs for. Each tuple contains (min_lon, min_lat, max_lon, max_lat).
    ref_img_path : str
        Path to the reference VRT file containing metadata (used for projection and extent info).
    resolution : float
        Desired output resolution for the DEM.
    Ancillary_folder : bool, optional
        If True, saves files in an auxiliary_folder_path. Defaults to False.
    buffer : float, optional
        Buffer to add around the DEM download area. Defaults to 0.02.
    Returns
    -------
    final_dem : str
        Path to the final DEM file.
    """
    
    # Define output paths for the DEM based on whether the Ancillary folder is used
    
       
    final_dem = os.path.join(auxiliary_folder_path, os.path.basename(os.path.dirname(os.path.dirname(ref_img_path))) + "_DEM.tif")
    output_dem = os.path.join(auxiliary_folder_path, "output_dem.tif")
    
    
    # Read the reference VRT file and extract projection and extent info
    area_info = open_image(ref_img_path)[1]  # Function open_image assumed to return metadata
    with rasterio.open(ref_img_path, 'r+') as rds:
        epsg_code = str(rds.crs).split(':')[1]
        
    # If DEM doesn't already exist, start the downloading process
    if not os.path.exists(final_dem):
        for idx, sub_area in enumerate(sub_areas):
                        
            temp_file = os.path.join(auxiliary_folder_path, f"dem_temp{idx}.tif")
            output_file = os.path.join(auxiliary_folder_path, f"dem_{idx}.tif")
           
            # If output file for the sub-area doesn't already exist, download and process the DEM
            if not os.path.exists(output_file):
                
                if epsg_code == "4326":  # WGS 84, no reprojection needed
                    min_lon, min_lat, max_lon, max_lat = sub_area
                else:
                    # Reproject the bounding box if not in WGS 84
                    srIn = osr.SpatialReference(str(area_info['projection']))

                    # Directly assign EPSG 4326 to srOut
                    srOut = osr.SpatialReference()
                    srOut.ImportFromEPSG(4326)
                    
                    # Reproject the extents from the input area
                    (min_lat, min_lon) = reproj_point(area_info["extent"][0], area_info["extent"][1], srIn, srOut)
                    (max_lat, max_lon) = reproj_point(area_info["extent"][2], area_info["extent"][3], srIn, srOut)
                    
                    resolution /= 100000  # Adjust resolution to match the projection
                    
                # Download the DEM with a buffer around the bounding box
                if not os.path.exists(temp_file):
                    elevation.clip(bounds=(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer), output=temp_file)
                    
                    # Reproject and resample the DEM if needed
                    if epsg_code == "4326":
                        cmd = f'gdalwarp -t_srs EPSG:{epsg_code} -te {min_lon} {min_lat} {max_lon} {max_lat} ' \
                              f'-r bilinear -tr {resolution} {resolution} {temp_file} {output_file}'
                    else:
                        # Apply reprojection using the original extents
                        E_min_old, N_min_old, E_max_old, N_max_old = area_info["extent"]
                        cmd = f'gdalwarp -t_srs EPSG:{epsg_code} -te {E_min_old} {N_min_old} {E_max_old} {N_max_old} ' \
                              f'-r bilinear -tr {resolution * 100000} {resolution * 100000} {temp_file} {output_file}'
                    # Run the command and remove temporary files
                    os.system(cmd)
                    os.remove(temp_file)
            else:
                print(f"{temp_file} already exists.")
        
        # If multiple sub-areas, merge them into a single DEM
        files_to_mosaic = sorted(glob.glob(os.path.dirname(output_file) + os.sep + "dem_*"))
        print(files_to_mosaic)
        if len(files_to_mosaic) > 1:
            files_string = " ".join(files_to_mosaic)
            cmd = f'gdal_merge.py -o {output_dem} {files_string}'
            os.system(cmd)
            
            # Apply a median filter to smooth the final DEM
            dem_data, dem_info = open_image(output_dem)
            smoothed_dem_data = median_filter(dem_data, 5)  # Apply 5x5 median filter
            
            # Save the smoothed DEM
            save_image(smoothed_dem_data, final_dem, 'GTiff', 6, dem_info['geotransform'], dem_info['projection'])
            # Clean up intermediate files
            for file in files_to_mosaic:
                os.remove(file)
            os.remove(output_dem)
        else:
            # If only one DEM file exists, rename it as the final DEM
            os.rename(files_to_mosaic[0], final_dem)
    else:
        print("DEM already exists.")
    
    return final_dem
def crop_predefined_DEM(ref_img_path, External_Dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False):
    """
    Crop a predefined DEM to match the spatial extent of a reference image and save it to the auxiliary folder.

    Parameters
    ----------
    ref_img_path : str
        Path to the reference image that defines the target spatial extent and projection.
    External_Dem_path : str
        Path to the external DEM file that will be cropped.
    auxiliary_folder_path : str
        Path to the auxiliary folder where the cropped DEM will be saved.
    reproj_type : str, optional
        GDAL resampling method for reprojection (e.g., 'bilinear', 'cubic'). The default is 'cubic'.
    overwrite : bool, optional
        If True, overwrite existing DEM. The default is False.

    Returns
    -------
    dem_path : str
        Path to the cropped DEM file.
    """

    # Define the path where the cropped DEM will be saved
    dem_path = os.path.join(auxiliary_folder_path, os.path.basename(os.path.dirname(os.path.dirname(ref_img_path))) + "_DEM.tif")
    
    # Check if the DEM file already exists and overwrite is set to False
    if os.path.exists(dem_path) and not overwrite:
        print('DEM file was already created and saved.')
        return dem_path
    
    # Open the reference image to get the spatial information (extent and projection)
    ref_img = gdal.Open(ref_img_path)
    if ref_img is None:
        raise FileNotFoundError(f"Reference image not found: {ref_img_path}")
    
    # Get the projection and extent information from the reference image
    ref_proj = osr.SpatialReference(wkt=ref_img.GetProjection())
    epsg_code = ref_proj.GetAttrValue("AUTHORITY", 1)  # Get EPSG code

    # Get the geospatial extent of the reference image (min/max coordinates)
    ref_gt = ref_img.GetGeoTransform()  # Get geotransform of the image
    E_min = ref_gt[0]  # Upper-left corner x-coordinate (min longitude/easting)
    N_min = ref_gt[3] + ref_gt[5] * ref_img.RasterYSize  # Lower-left corner y-coordinate (min latitude/northing)
    E_max = ref_gt[0] + ref_gt[1] * ref_img.RasterXSize  # Upper-right corner x-coordinate (max longitude/easting)
    N_max = ref_gt[3]  # Upper-left corner y-coordinate (max latitude/northing)
    
    # Define the output resolution (same as input DEM)
    dem_resolution = ref_gt[1]  # Pixel resolution (in meters or degrees)

    # Build the gdalwarp command to crop the DEM to the reference image extent and reproject if necessary
    cmd = f'gdalwarp -t_srs EPSG:{epsg_code} -te {E_min} {N_min} {E_max} {N_max} ' \
          f'-r {reproj_type} -tr {dem_resolution} {dem_resolution} {External_Dem_path} {dem_path}'
    
    # Run the gdalwarp command
    os.system(cmd)
    
    # Return the path to the newly created DEM
    return dem_path

def calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False):
    '''
    Calculate slope and aspect from an input DEM.

    Parameters
    ----------
    dem_path : str
        Path to the existing DEM file.
    outdir : str
        Output directory where the slope and aspect files will be saved.
    resolution : float
        Output resolution.
    reproj_type : str, optional
        GDAL resampling method for reprojection (e.g., 'bilinear', 'cubic'). The default is 'cubic'.
    overwrite : bool, optional
        If True, overwrite existing slope and aspect files. The default is False.
    Ancillary_folder : bool, optional
        If True, save files in an ancillary folder within the output directory. The default is False.

    Returns
    -------
    slopePath : str
        Path to the saved slope file.
    aspectPath : str
        Path to the saved aspect file.
    '''

    
    slopePath = os.path.join(auxiliary_folder_path, os.path.basename(dem_path).replace('_DEM.tif', '_slope.tif'))
    aspectPath = os.path.join(auxiliary_folder_path, os.path.basename(dem_path).replace('_DEM.tif', '_aspect.tif'))
    
    print(slopePath)

    ################### Calculate slope
    if os.path.exists(slopePath) and not overwrite:
        print('Slope file was already created and saved')
    else:
        cmd = f"gdaldem slope {dem_path} {slopePath} -of GTiff -compute_edges"
        os.system(cmd)
        print(f"Slope saved at {slopePath}")
        
    ################### Calculate aspect
    if os.path.exists(aspectPath) and not overwrite:
        print('Aspect file was already created and saved')
    else:
        cmd = f"gdaldem aspect {dem_path} {aspectPath} -of GTiff -compute_edges"
        os.system(cmd)
        print(f"Aspect saved at {aspectPath}")

    return slopePath, aspectPath

def S2_clouds_classifier(stack_clouds_path, path_cloud_mask, ref_img_path, cloud_prob, overwrite_cloud = 0, average_over=2, dilation_size=3):
    
    from s2cloudless import S2PixelCloudDetector
    """
    Classifies clouds in a Sentinel-2 image.

    Args:
        stack_clouds_path: Path to the stack of cloud bands.
        ref_img_path: Path to the stack of SCF bands.
        cloud_prob: Cloud probability threshold.
        overwrite_cloud: Whether to overwrite the existing cloud mask.
        average_over: Size of the averaging window.
        dilation_size: Size of the dilation operation.

    Returns:
        path_cloud_mask: Path to the generated cloud mask.
        cloud_cover_percentage: Cloud cover percentage.
    """

   
    temporary_cloud_mask_path = path_cloud_mask.replace('.tif', '60m.tif') 
    
    if not os.path.exists(temporary_cloud_mask_path):
        
        input_features, info = open_image(stack_clouds_path)
        Stack_to_classify = np.zeros((1, info['X_Y_raster_size'][1], info['X_Y_raster_size'][0], 10))
        input_features = np.transpose(input_features, (1, 2, 0))
        Stack_to_classify[0, :, :, :] = input_features
        
        Stack_to_classify[0, :, :, :] [input_features[:,:,0] == 255]=0
        print('Bands for cloud classification ready...')

        try:
            cloud_detector = S2PixelCloudDetector(threshold=cloud_prob, average_over=average_over, dilation_size=dilation_size)
            cloud_mask = cloud_detector.get_cloud_masks(np.array(Stack_to_classify)) + 1
            
            cloud_cover_percentage = np.sum(cloud_mask[0, :, :] == 2) / \
                (np.shape(cloud_mask[0, :, :])[0] * np.shape(cloud_mask[0, :, :])[1])
           
            save_image(cloud_mask[0, :, :], temporary_cloud_mask_path, 'GTiff', 1, info['geotransform'],
                       info['projection'])
        except Exception as e:
            print(f"Error during cloud classification: {e}")
            # Handle the error appropriately, e.g., log it or raise an exception

    if not os.path.exists(path_cloud_mask) or overwrite_cloud==1:
        
        tgt_img_info = open_image(ref_img_path)[1]
        
        E_min = (tgt_img_info['extent'][0])
        N_min = (tgt_img_info['extent'][1])
        E_max = (tgt_img_info['extent'][2])
        N_max = (tgt_img_info['extent'][3])
        img_res = str(tgt_img_info['geotransform'][1])
     
        cmd = 'gdalwarp -te ' + ' '.join((str(E_min), str(N_min), str(E_max), str(N_max))) +\
            ' -r nearest -tr ' + ' '.join((str(img_res), str(img_res))) + ' ' + ' '.join((temporary_cloud_mask_path, path_cloud_mask))
        os.system(cmd)
        clud_tot=open_image( path_cloud_mask)[0]
        
        cloud_cover_percentage = np.sum(clud_tot[ :, :] == 2) / \
            (np.shape(clud_tot[ :, :])[0] * np.shape(clud_tot[ :, :])[1])
            
        os.remove(temporary_cloud_mask_path) 
    else:
        cloud_mask = open_image(path_cloud_mask)[0]
        
        cloud_cover_percentage = np.sum(cloud_mask == 2) / \
            (np.shape(cloud_mask)[0] * np.shape(cloud_mask)[1])
    
    return path_cloud_mask, cloud_cover_percentage;
    
def create_default_cloud_mask(shape, path_cloud_mask):
    
    path_cloud_mask = Path(path_cloud_mask)
    parent_one_levels_up = path_cloud_mask.parents[1]
    ref_img_path = glob.glob(os.path.join(parent_one_levels_up, '*scf.vrt'))
    
    if ref_img_path == []:
        ref_img_path = glob.glob(os.path.join(parent_one_levels_up, 'PRS*.tif'))[0]
    else:
        ref_img_path = ref_img_path[0]
        
    img_info = open_image(ref_img_path)[1]
    cloud_mask = np.zeros_like(shape) + 1
    save_image(cloud_mask, path_cloud_mask, 'GTiff', 1, img_info['geotransform'], img_info['projection'])
    del cloud_mask   
      
def generate_no_data_mask(L_image, sensor, no_data_value=np.nan):
    """
    Generates a no-data mask for a given image based on the sensor.

    Args:
        L_image: The input image as a NumPy array.
        sensor: The sensor type (e.g., "L5", "L7").
        no_data_value: The value representing no data in the image.

    Returns:
        The generated no-data mask as a NumPy boolean array.
    """

    if np.isnan(no_data_value):
        if sensor == "L5":
            no_data_mask = (np.isnan(L_image[5, :, :]) | np.isnan(L_image[0, :, :])).astype(bool)
        elif sensor == "L7":
            no_data_mask = (np.isnan(L_image[0, :, :]) | np.isnan(L_image[np.max(np.shape(L_image)[0] - 1), :, :]) | np.isnan(L_image[6, :, :])).astype(bool)
        else:
            #no_data_mask = (np.isnan(L_image[0, :, :]) | np.isnan(L_image[np.max(np.shape(L_image)[0] - 1), :, :])).astype(bool)
            no_data_mask = np.any(np.isnan(L_image), axis=0)
    else:
        # Handle other no-data values if needed
        raise NotImplementedError("Handling of non-NaN no-data values is not implemented yet.")
        
    valid_mask = np.logical_not(no_data_mask)

    return no_data_mask, valid_mask
    


    
def spectral_idx_computer(B1, B2, idx_name, curr_image, no_data_mask, curr_aux_folder, sensor, output_filename, ref_img_path, B3=None, B4=None):
    """
    Computes a spectral index and saves the result in the specified folder.
    
    Parameters
    ----------
    B1, B2 : numpy.ndarray
        Input bands used to calculate the spectral index.
        
    idx_name : str
        Name of the spectral index (e.g., 'NDSI', 'NDVI', 'shad_idx').
        
    curr_image : numpy.ndarray
        Original 3D image data.
        
    no_data_mask : numpy.ndarray
        Mask indicating no-data values.
        
    curr_aux_folder : str
        Path to the folder where the output will be saved.
        
    sensor : str
        Type of sensor.
        
    output_filename : str
        Name of the output file.
        
    ref_img_path : str
        Path to the reference image to obtain metadata.
    
    Returns
    -------
    numpy.ndarray
        Computed spectral index.
    """
    
    # Define the calculations for each index
    calculations = {
        'normDiff': lambda B1, B2, B3, B4: (B1 - B2) / (B1 + B2),
        'shad_idx': lambda B1, B2, B3, B4:  (B1 - B2) / (B1 + B2) / B1,
        'band_diff': lambda B1, B2, B3, B4: B1-B2,
        'EVI': lambda B1, B2, B3, B4: 2.5*(B1-B2) / (B1+2.4*B2+1),
        'NDSIplus': lambda B1, B2, B3, B4: 2*(B1+B2-B3-B4) / (B1+B2+B3+B4),
        'idx6': lambda B1, B2, B3, B4: 2*(2*B1-B2-B3) / (2*B1+B2+B3)
        
        
    }
    
    # Check if the index name is in the dictionary
    if idx_name not in calculations:
        raise ValueError(f"Index '{idx_name}' is not supported.")
    
    
    # Perform the calculation
    idx_out = calculations[idx_name](B1, B2, B3, B4)
    
    # Set the pixels corresponding to no_data_mask to invalid (e.g., np.nan)
    idx_out[no_data_mask] = np.nan
    
    # Save the computed band to a file in curr_aux_folder
    output_path = os.path.join(curr_aux_folder, output_filename)
    
    # Open the reference image to get metadata
    with rasterio.open(ref_img_path) as src:
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': idx_out.shape[0],
            'width': idx_out.shape[1],
            'count': 1,
            'dtype': idx_out.dtype,
            'nodata': np.nan
        })
        
        # Write the spectral index to a new file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(idx_out, 1)
    
    print(f"Spectral index {idx_name} saved at {output_path}")
    return ;


def solar_incidence_angle_calculator(curr_image_info, date_time, slopePath, aspectPath, curr_aux_folder, date):
    """
    Calculates the solar incidence angle based on slope, aspect, sun altitude, and azimuth.

    Parameters
    ----------
    img_info : dict
        Dictionary containing image metadata such as extent, geotransform, and EPSG code.

    date_time : datetime
        Date and time for which the solar position is calculated.

    slope_path : str
        Path to the slope GeoTIFF.

    aspect_path : str
        Path to the aspect GeoTIFF.

    curr_aux_folder : str
        Path to the folder where the solar incidence angle result will be saved.

    Returns
    -------
    numpy.ndarray
        Array representing the solar incidence angle.
    """
    # Extract image metadata
    E_min, N_min, E_max, N_max = curr_image_info['extent']
    epsg_code = curr_image_info['EPSG']

    # Transform the coordinates to WGS84
    transformer = Transformer.from_crs(f"epsg:{epsg_code}", "epsg:4326", always_xy=True)
    central_E = E_min + (E_max - E_min) / 2
    central_N = N_min + (N_max - N_min) / 2
    Central_WGS84 = transformer.transform(central_E, central_N)

    # Convert date_time to UTC timezone
    datetime_object = date_time.replace(tzinfo=timezone.utc)

    # Get sun altitude and azimuth
    sun_altitude = get_altitude(Central_WGS84[1], Central_WGS84[0], datetime_object)
    sun_azimuth = get_azimuth(Central_WGS84[1], Central_WGS84[0], datetime_object)

    # Convert angles from degrees to radians
    sun_zenith_rad = np.radians(90 - sun_altitude)
    sun_azimuth_rad = np.radians(sun_azimuth)

    # Read slope and aspect from the files
    with rasterio.open(slopePath) as slope_ds, rasterio.open(aspectPath) as aspect_ds:
        slope = slope_ds.read(1)
        aspect = aspect_ds.read(1)
        profile = slope_ds.profile

    # Convert slope and aspect from degrees to radians
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)

    # Calculate the solar incidence angle
    solar_incidence_angle = np.degrees(np.arccos(
        np.cos(sun_zenith_rad) * np.cos(slope_rad) +
        np.sin(sun_zenith_rad) * np.sin(slope_rad) * np.cos(aspect_rad - sun_azimuth_rad)
    ))

    # Set no-data areas (where slope is invalid) to NaN
    solar_incidence_angle[np.isnan(slope)] = np.nan

    # Save the solar incidence angle to a GeoTIFF in the curr_aux_folder
    output_path = os.path.join(curr_aux_folder, date + '_solar_incidence_angle.tif')
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(solar_incidence_angle.astype(np.float32), 1)

    print(f"Solar incidence angle saved at {output_path}")
    return solar_incidence_angle      
    
def generate_shadow_mask(curr_aux_folder, auxiliary_folder_path, no_data_mask, NIR):
    """
    Generate a shadow mask dynamically without setting thresholds and save as GeoTIFF.

    Parameters:
    - curr_aux_folder: Path to the auxiliary folder containing GeoTIFF files for indices.
    """
    # Find the paths to the necessary GeoTIFF files
    ndvi_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    idx6_path = glob.glob(os.path.join(curr_aux_folder, '*idx6.tif'))[0]
    evi_path = glob.glob(os.path.join(curr_aux_folder, '*EVI.tif'))[0]
    shad_idx_path = glob.glob(os.path.join(curr_aux_folder, '*shad_idx.tif'))[0]
    
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    
    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]
    
    
    # Read input GeoTIFFs
    with rasterio.open(idx6_path) as src1, \
         rasterio.open(shad_idx_path) as src2, \
         rasterio.open(ndvi_path) as src_ndvi, \
         rasterio.open(evi_path) as src_evi, \
         rasterio.open(path_cloud_mask) as src_clouds, \
         rasterio.open(path_water_mask) as src_water, \
         rasterio.open(solar_incidence_angle_path) as src_angle:
        
        # Read data arrays
        index1 = src1.read(1).astype(float)
        index2 = src2.read(1).astype(float)
        ndvi = src_ndvi.read(1).astype(float)
        evi = src_evi.read(1).astype(float)
        cloud_mask = src_clouds.read(1).astype(int)
        water_mask = src_water.read(1).astype(int)
        solar_incidence_angle = src_angle.read(1).astype(float)
        
        # Read metadata for output
        meta = src1.meta.copy()
        
    # Normalize indices to range [0, 1]
    def normalize(arr):
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        return (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)
    
    curr_range = (90, 180)   
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))
    curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0], solar_incidence_angle < curr_range[1]))
    
    
    index1_norm = normalize(index1)
    index2_norm = normalize(index2)
    ndvi_norm = normalize(ndvi)
    evi_norm = normalize(evi)
    
    
    # Combine indices to create a composite shadow score
    # Shadow pixels maximize index1 and index2, minimize ndvi and evi
    shadow_score = (index1_norm + index2_norm) - (ndvi_norm + evi_norm + NIR)
    
    threshold = np.percentile(shadow_score[curr_angle_valid], [5, 95])[0]
    #plt.hist(shadow_score.flatten(), bins=50)
    # Create shadow mask: positive values indicate shadow
    shadow_mask = (shadow_score > threshold).astype(np.uint8)
    
    # Update metadata for output
    meta.update({
        "dtype": "uint8",
        "count": 1,
        "nodata": 255,  # Use 255 as the nodata value for uint8
        "compress": "lzw"  # Compression to reduce file size
    })
    
    # Save shadow mask to GeoTIFF
    output_path = os.path.join(curr_aux_folder, 'shadow_mask.tif')
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(shadow_mask, 1)
    
    print(f"Shadow mask saved to {output_path}")
    
    
    