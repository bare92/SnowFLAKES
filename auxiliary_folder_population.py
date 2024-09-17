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
        
        d = gdal.Open(ref_img_path)
        proj = osr.SpatialReference(wkt=d.GetProjection())
        epsg_code = proj.GetAttrValue("AUTHORITY", 1)   
        
        resolution=dim_img['geotransform'][1]
        E_min=dim_img["extent"][0]
        N_min=dim_img["extent"][1]
        E_max=dim_img["extent"][2]
        N_max=dim_img["extent"][3]
        
        if epsg_code!="4326":
            srIn = osr.SpatialReference(str(dim_img['projection']))
            file=glob.glob("/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/*")[0]
            d_target = open_image(file)[1]
            srOut = osr.SpatialReference(str(d_target['projection']))
            E_min_old=dim_img["extent"][0]
            N_min_old=dim_img["extent"][1]
            E_max_old=dim_img["extent"][2]
            N_max_old=dim_img["extent"][3]
            (N_min, E_min) = reproj_point(E_min_old, N_min_old, srIn, srOut)
            (N_max, E_max) = reproj_point(E_max_old, N_max_old, srIn, srOut)
            resolution=resolution/100000
                    
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
    if Ancillary_folder:
        target_wb_mask_path = auxiliary_folder_path+os.sep+os.path.basename(os.path.dirname( os.path.dirname(ref_img_path)))+ "_Water_Mask.tif"  
    else:
        target_wb_mask_path = ref_img_path[:-8] + "_Water_Mask.tif"  
    
    if not os.path.exists(target_wb_mask_path):
    # clip the wbm with FSC extent
      
        img_info = open_image(ref_img_path)[1]
    
        d = gdal.Open(ref_img_path)
        proj = osr.SpatialReference(wkt=d.GetProjection())
        epsg_code_ref = proj.GetAttrValue("AUTHORITY", 1)
            
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
        global_water_mask = glob.glob("/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/*")[0]
        srOut = osr.SpatialReference(str(open_image(global_water_mask)[1]['projection']))
        
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
    d = gdal.Open(ref_img_path)
    proj = osr.SpatialReference(wkt=d.GetProjection())
    epsg_code = proj.GetAttrValue("AUTHORITY", 1)  # Extract EPSG code

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
                    file = glob.glob("/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/*")[0]
                    d_target = open_image(file)[1]
                    srOut = osr.SpatialReference(str(d_target['projection']))

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

def S2_clouds_classifier(stack_clouds_path, stack_fsc_path, cloud_prob, overwrite_cloud, average_over=2, dilation_size=3):
    """
    Classifies clouds in a Sentinel-2 image.

    Args:
        stack_clouds_path: Path to the stack of cloud bands.
        stack_fsc_path: Path to the stack of SCF bands.
        cloud_prob: Cloud probability threshold.
        overwrite_cloud: Whether to overwrite the existing cloud mask.
        average_over: Size of the averaging window.
        dilation_size: Size of the dilation operation.

    Returns:
        path_cloud_mask: Path to the generated cloud mask.
        cloud_cover_percentage: Cloud cover percentage.
    """

    path_cloud_mask = stack_clouds_path[:-4] + '_Mask.tif'
   
    temporary_cloud_mask_path = stack_clouds_path[:-4] + '_Mask_60m.tif'
    
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
        
        tgt_img_info = open_image(stack_fsc_path)[1]
        
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
    
def create_default_cloud_mask(shape, path):
    cloud_mask = np.zeros_like(shape) + 1
    save_image(cloud_mask, path, 'GTiff', 1, img_info['geotransform'], img_info['projection'])
    del cloud_mask   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    