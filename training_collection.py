#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:07:46 2024

@author: rbarella
"""

from utilities import *
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def read_masked_values(geotiff_path, mask, band = 1):
    """
    Reads the values of a GeoTIFF corresponding to a logical mask.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTIFF file.
    mask : numpy.ndarray
        A 2D boolean mask (True where you want to keep values, False otherwise).

    Returns
    -------
    masked_values : numpy.ndarray
        Array of values corresponding to the True positions in the mask.
    """
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Read the data
        data = src.read(band)  # Read the first band (change the index for other bands if needed)

        # Apply the mask to get only the values where mask is True
        masked_values = data[mask]

    return masked_values


def collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, bands):
    
    sensor = get_sensor(os.path.basename(curr_acquisition))
    
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    
    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))[0]
    
    bands = define_bands(curr_image, valid_mask, sensor)
    
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    solar_incidence_angle = open_image(solar_incidence_angle_path)[0]
    
    
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))
    
    training_samples_df = pd.DataFrame(columns=['no_snow', 'snow', 'shadow', 'NDSI'])
    
    ranges = ((0,20), (20, 45), (45, 70), (70, 90), (90, 180))
    
    for curr_range in ranges:
    
        curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0], solar_incidence_angle < curr_range[1]))
              
        curr_NDSI = read_masked_values(NDSI_path, curr_angle_valid)
        
        curr_NDVI = read_masked_values(NDVI_path, curr_angle_valid)
        
        curr_green = read_masked_values(bands_path, curr_angle_valid, band = 2)
    
        plt.hist(curr_green, bins=100, alpha=0.5, label= str(curr_range))
        
        features = np.stack([curr_NDSI, curr_NDVI, curr_green], axis=-1)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_components = 3  # Number of clusters, adjust as needed
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm_labels = gmm.fit_predict(features_scaled)
        
        cluster_means = []
        for cluster in range(n_components):
            cluster_data = features[gmm_labels == cluster]
            mean_ndsi = cluster_data[:, 0].mean()
            mean_ndvi = cluster_data[:, 1].mean()
            mean_green = cluster_data[:, 2].mean()
            cluster_means.append((mean_ndsi, mean_ndvi, mean_green))

        # Select the cluster that maximizes NDSI and green reflectance, and minimizes NDVI
        selected_cluster = np.argmax([ndsi - ndvi + green for ndsi, ndvi, green in cluster_means])
        
        # Step 5: Create a mask for the selected cluster
        selected_mask = (gmm_labels == selected_cluster)
        
        empty = np.zeros(curr_angle_valid.shape)
        empty[curr_angle_valid] = gmm_labels + 1
        
        output_path = os.path.join(curr_aux_folder, 'clusters_for_endmember.tif')
        
        # Use the profile from one of the input rasters (e.g., NDSI) for metadata
        with rasterio.open(NDSI_path) as src:
            profile = src.profile
        
        # Update the profile for the output raster
        profile.update(dtype='uint8', count=1, compress='lzw', nodata=0)
        
        # Save the selected mask as a new GeoTIFF file with 0 as nodata
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(empty.astype('uint8'), 1)
        
        plt.figure()
        plt.imshow(empty)
    
    plt.legend()
    
    




