#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:07:46 2024

@author: rbarella
"""

from utilities import *
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import binary_erosion
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import distance
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist

def read_masked_values(geotiff_path, mask, bands=None):
    """
    Reads the values of a multispectral GeoTIFF corresponding to a logical mask.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTIFF file.
    mask : numpy.ndarray
        A 2D boolean mask (True where you want to keep values, False otherwise).
    bands : list of int, optional
        List of band indices to read (1-based index). If None, all bands are read.

    Returns
    -------
    masked_values : numpy.ndarray
        2D array of values where each row contains the pixel values across bands 
        for locations where the mask is True.
    """
    with rasterio.open(geotiff_path) as src:
        # If bands are not specified, read all bands
        if bands is None:
            bands = list(range(1, src.count + 1))
        
        # List to store masked values for each band
        masked_values_per_band = []

        for band in bands:
            data = src.read(band)  # Read each specified band
            masked_values_per_band.append(data[mask])  # Apply mask and store result

        # Stack the results to create a 2D array with shape (num_pixels, num_bands)
        masked_values = np.stack(masked_values_per_band, axis=-1)

    return masked_values


def get_representative_pixels(bands_data, valid_mask, k=5, n_closest=5):
    """
    Selects representative "no snow" pixels by clustering and distance to cluster centroids.
    Saves the output as a raster.

    Parameters
    ----------
    bands_data : numpy.ndarray
        3D array (bands, height, width) containing spectral data for each band.
    valid_mask : numpy.ndarray
        2D mask of valid pixels for selection.
    NDSI_path : str
        Path to the NDSI file for extracting the profile.
    output_path : str
        Path to save the representative pixels mask as a GeoTIFF.
    k : int, optional
        Number of clusters for K-means, by default 5.
    n_closest : int, optional
        Number of closest pixels to each centroid to select, by default 10.

    Returns
    -------
    None
    """

    # Extract "no snow" pixels for clustering
    no_snow_pixels = bands_data[valid_mask, :]  # Shape (pixels, bands)
    
    # Perform K-means clustering on "no snow" pixels
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(no_snow_pixels)
    
    # Get cluster centroids and labels
    #labels, centroids = kernel_kmeans(no_snow_pixels, k, gamma)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

   
    representative_pixels_mask = np.zeros(valid_mask.shape, dtype='uint8')
    
    # Find the n_closest pixels to each centroid
    for cluster_idx in range(k):
        # Select pixels in the current cluster
        cluster_pixels = no_snow_pixels[labels == cluster_idx]
        
        # Compute distances to the centroid for these pixels
        distances = distance.cdist(cluster_pixels, [centroids[cluster_idx]], 'euclidean').flatten()
        
        # Get the indices of the n_closest pixels in the cluster
        closest_indices = np.argsort(distances)[:n_closest]
        
        # Map the closest indices back to the original image coordinates
        original_indices = np.argwhere(valid_mask)[labels == cluster_idx]
        selected_pixels = original_indices[closest_indices]
        
        # Set these pixels in the representative mask
       
        representative_pixels_mask[selected_pixels] = 1

    return representative_pixels_mask;
 
def collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path, SVM_folder_name, no_data_mask, bands, PCA=False):
    
    scf_folder = os.path.join(curr_acquisition, SVM_folder_name)
    if not os.path.exists(scf_folder):
        os.makedirs(scf_folder)
        
    sensor = get_sensor(os.path.basename(curr_acquisition))
    
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    diff_B_NIR_path = glob.glob(os.path.join(curr_aux_folder, '*diffBNIR.tif'))[0]
    shad_idx_path = glob.glob(os.path.join(curr_aux_folder, '*shad_idx.tif'))[0]
        
    
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
    
    if bands_path == []:
        bands_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
    else:
        bands_path = bands_path[0]
        
    valid_mask = np.logical_not(no_data_mask)
    
    # Load masks and other necessary data
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    solar_incidence_angle = open_image(solar_incidence_angle_path)[0]
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))
    
    ranges = ((0,20), (20, 45), (45, 70), (70, 90), (90, 180))
    
    #ranges = ((70, 90))
    
    #ranges = ((0,20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 180))
    empty = np.zeros(curr_scene_valid.shape, dtype='uint8')
    
    for curr_range in ranges:
        print(curr_range)
        
        # Initialize as empty arrays
        representative_pixels_mask_snow = np.array([])
        representative_pixels_mask_noSnow = np.array([])
    
        curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0], solar_incidence_angle < curr_range[1]))
    
        curr_NDSI = read_masked_values(NDSI_path, curr_angle_valid)
        curr_NDVI = read_masked_values(NDVI_path, curr_angle_valid)
        curr_green = read_masked_values(bands_path, curr_angle_valid, bands=[2])
        curr_bands = read_masked_values(bands_path, curr_angle_valid)
        curr_diff_B_NIR = read_masked_values(diff_B_NIR_path, curr_angle_valid)
        curr_shad_idx = read_masked_values(shad_idx_path, curr_angle_valid)
    
        # SNOW TRAINING
        if curr_range[0] >= 90:
            # Normalize indices and compute shadow metric
            diff_B_NIR_low_perc, diff_B_NIR_high_perc = np.percentile(curr_diff_B_NIR, [2, 98])
            shad_idx_low_perc, shad_idx_high_perc = np.percentile(curr_shad_idx, [2, 98])
            curr_diff_B_NIR_norm = np.clip((curr_diff_B_NIR - diff_B_NIR_low_perc) / (diff_B_NIR_high_perc - diff_B_NIR_low_perc), 0, 1)
            curr_shad_idx_norm = np.clip((curr_shad_idx - shad_idx_low_perc) / (shad_idx_high_perc - shad_idx_low_perc), 0, 1)
            curr_score_snow_shadow = curr_diff_B_NIR_norm - curr_shad_idx_norm
            threshold_shadow = np.percentile(curr_score_snow_shadow, 98)
            curr_valid_snow_mask_shadow = np.logical_and(curr_score_snow_shadow >= threshold_shadow, curr_NDSI > 0.7).flatten()
            if np.sum(curr_valid_snow_mask_shadow) > 10:
                representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask_shadow, k=3, n_closest=10)
        else:
            # Normalize indices and compute sun metric
            NDSI_low_perc, NDSI_high_perc = np.percentile(curr_NDSI, [1, 99])
            NDVI_low_perc, NDVI_high_perc = np.percentile(curr_NDVI, [1, 99])
            green_low_perc, green_high_perc = np.percentile(curr_green, [1, 99])
            curr_NDSI_norm = np.clip((curr_NDSI - NDSI_low_perc) / (NDSI_high_perc - NDVI_low_perc), 0, 1)
            curr_NDVI_norm = np.clip((curr_NDVI - NDVI_low_perc) / (NDVI_high_perc - NDVI_low_perc), 0, 1)
            curr_green_norm = np.clip((curr_green - green_low_perc) / (green_high_perc - green_low_perc), 0, 1)
            curr_score_snow_sun = curr_NDSI_norm - curr_NDVI_norm + curr_green_norm
            threshold = np.percentile(curr_score_snow_sun, 95)
            curr_valid_snow_mask = np.logical_and(curr_score_snow_sun >= threshold, curr_NDSI > 0.7).flatten()
            
            if np.sum(curr_valid_snow_mask) > 10:
                representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask, k=5, n_closest=10)
    
        ## NO snow TRAINING
        if curr_range[0] >= 90:
            threshold_shadow_no_snow = np.percentile(curr_score_snow_shadow, 5)
            curr_valid_no_snow_mask_shadow = (curr_score_snow_shadow <= threshold_shadow_no_snow).flatten()
            
            if np.sum(curr_valid_no_snow_mask_shadow) > 10:
                representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask_shadow, k=3, n_closest=10) * 2
        else:
            curr_valid_no_snow_mask = (curr_NDSI < 0).flatten()
            
            if np.sum(curr_valid_no_snow_mask) > 10:
                representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask, k=10, n_closest=5) * 2
    
        # Check if masks have been assigned; if not, set as zeros
        if representative_pixels_mask_snow.size == 0:
            representative_pixels_mask_snow = np.zeros(curr_angle_valid.sum(), dtype='uint8')
        if representative_pixels_mask_noSnow.size == 0:
            representative_pixels_mask_noSnow = np.zeros(curr_angle_valid.sum(), dtype='uint8')
            
        representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
        empty[curr_angle_valid] = representative_pixels_mask
        
        print(str(np.sum(representative_pixels_mask_snow.flatten())) + ' SNOW PIXELS')
        print(str(np.sum(representative_pixels_mask_noSnow.flatten() / 2)) + ' NO SNOW PIXELS')


    # Convert points where result == 1 or 2 to a shapefile
    points = []
    values = []
    with rasterio.open(NDSI_path) as src:
        for row, col in zip(*np.where((empty == 1) | (empty == 2))):
            x, y = src.xy(row, col)
            points.append(Point(x, y))
            values.append(empty[row, col])

    gdf = gpd.GeoDataFrame({"value": values}, geometry=points, crs=src.crs)
    svm_folder_path = os.path.join(curr_acquisition, SVM_folder_name)
    shapefile_path = os.path.join(svm_folder_path, 'representative_pixels_for_training_samples.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")
    
    training_mask_path = os.path.join(svm_folder_path, 'representative_pixels_for_training_samples.tif')
    
    # Update the profile and save the representative mask
    with rasterio.open(NDSI_path) as src:
        profile = src.profile
    profile.update(dtype='uint8', count=1, compress='lzw', nodata=0)
    
    with rasterio.open(training_mask_path, 'w', **profile) as dst:
        dst.write(empty, 1)

    return shapefile_path , training_mask_path   
  
def rbf_kernel(X, gamma):
    """
    Computes the RBF (Gaussian) kernel matrix for the dataset X.
    
    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data.
    - gamma: float
        The parameter for the RBF kernel, defined as 1 / (2 * sigma^2).
        
    Returns:
    - K: np.ndarray, shape (n_samples, n_samples)
        The RBF kernel matrix.
    """
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    K = np.exp(-gamma * pairwise_sq_dists)
    return K

def kernel_kmeans(X, n_clusters, gamma, max_iter=100):
    """
    Performs k-means clustering using an RBF kernel instead of Euclidean distance.
    
    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data.
    - n_clusters: int
        Number of clusters.
    - gamma: float
        Parameter for the RBF kernel, defined as 1 / (2 * sigma^2).
    - max_iter: int, optional, default=100
        Maximum number of iterations for the algorithm.
        
    Returns:
    - labels: np.ndarray, shape (n_samples,)
        The cluster labels for each sample.
    - centroids: np.ndarray, shape (n_clusters, n_features)
        The approximated centroids in the original feature space.
    """
    # Step 1: Compute the kernel matrix
    K = rbf_kernel(X, gamma)
    
    # Step 2: Initialize cluster assignments randomly
    n_samples = X.shape[0]
    labels = np.random.randint(0, n_clusters, n_samples)
    
    # Step 3: Kernel k-means iteration
    for _ in range(max_iter):
        # Compute distances to centroids in the transformed space
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            cluster_k = K[:, labels == k]
            N_k = np.sum(labels == k)
            if N_k > 0:
                # Compute the distance to the "centroid" in the kernel space
                distances[:, k] = np.diag(K) - 2 * np.sum(cluster_k, axis=1) / N_k + \
                                  np.sum(K[labels == k][:, labels == k]) / (N_k ** 2)
        
        # Update labels based on minimal distance
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.all(labels == new_labels):
            break
        
        labels = new_labels

    # Step 4: Compute approximate centroids in the original space
    centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
    
    return labels, centroids


