import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



def compute_representative_snow_pixels(curr_NDSI, curr_NDVI, curr_bands, curr_distance_idx, curr_green, sample_count):
    # Normalize indices and compute sun metric
    representative_pixels_mask_snow = np.array([])
    representative_pixels_mask_noSnow = np.array([])

    NDSI_low_perc, NDSI_high_perc = np.percentile(curr_NDSI[np.logical_not(np.isnan(curr_NDSI))], [1, 99])
    NDVI_low_perc, NDVI_high_perc = np.percentile(curr_NDVI[np.logical_not(np.isnan(curr_NDVI))], [1, 99])
    green_low_perc, green_high_perc = np.percentile(curr_green, [1, 99])
    curr_NDSI_norm = np.clip((curr_NDSI - NDSI_low_perc) / (NDSI_high_perc - NDVI_low_perc), 0, 1)
    curr_NDVI_norm = np.clip((curr_NDVI - NDVI_low_perc) / (NDVI_high_perc - NDVI_low_perc), 0, 1)
    curr_green_norm = np.clip((curr_green - green_low_perc) / (green_high_perc - green_low_perc), 0, 1)
    curr_score_snow_sun = curr_NDSI_norm - curr_NDVI_norm + curr_green_norm
    threshold = np.percentile(curr_score_snow_sun, 95)
    curr_valid_snow_mask = np.logical_and.reduce(
        (curr_score_snow_sun >= threshold, curr_NDSI > 0.7, curr_distance_idx != 255)).flatten()
    if np.sum(curr_valid_snow_mask) > 10:
        representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask,
                                                                    sample_count=int(sample_count / 2), k=5,
                                                                    n_closest='auto')
    curr_valid_no_snow_mask = (curr_NDSI < 0).flatten()
    if np.sum(curr_valid_no_snow_mask) > 10:
        representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask,
                                                                      sample_count=int(sample_count / 2), k=10,
                                                                      n_closest='auto') * 2
    return representative_pixels_mask_snow, representative_pixels_mask_noSnow


def compute_representative_snow_pixels_high_range(curr_NDSI, curr_bands, curr_diff_B_NIR, curr_distance_idx, curr_shad_idx,
                                                  sample_count):
    representative_pixels_mask_snow = np.array([])
    representative_pixels_mask_noSnow = np.array([])
    diff_B_NIR_low_perc, diff_B_NIR_high_perc = np.percentile(curr_diff_B_NIR, [2, 95])
    shad_idx_low_perc, shad_idx_high_perc = np.percentile(curr_shad_idx, [2, 95])
    curr_diff_B_NIR_norm = np.clip(
        (curr_diff_B_NIR - diff_B_NIR_low_perc) / (diff_B_NIR_high_perc - diff_B_NIR_low_perc), 0, 1)
    curr_shad_idx_norm = np.clip((curr_shad_idx - shad_idx_low_perc) / (shad_idx_high_perc - shad_idx_low_perc),
                                 0, 1)
    curr_score_snow_shadow = curr_diff_B_NIR_norm - curr_shad_idx_norm
    threshold_shadow = np.percentile(curr_score_snow_shadow, 95)
    curr_valid_snow_mask_shadow = np.logical_and.reduce(
        (curr_score_snow_shadow >= threshold_shadow, curr_NDSI > 0.7, curr_distance_idx != 255)).flatten()
    if np.sum(curr_valid_snow_mask_shadow) > 10:
        representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask_shadow,
                                                                    sample_count=int(sample_count / 2), k=5,
                                                                    n_closest='auto')

    threshold_shadow_no_snow = np.percentile(curr_score_snow_shadow, 5)
    curr_valid_no_snow_mask_shadow = (curr_score_snow_shadow <= threshold_shadow_no_snow).flatten()

    if np.sum(curr_valid_no_snow_mask_shadow) > 10:
        representative_pixels_mask_noSnow = get_representative_pixels(curr_bands,
                                                                      curr_valid_no_snow_mask_shadow,
                                                                      sample_count=int(sample_count / 2), k=5,
                                                                      n_closest='auto') * 2

    return representative_pixels_mask_snow, representative_pixels_mask_noSnow


def get_representative_pixels(bands_data, valid_mask, sample_count=50, k='auto', n_closest='auto'):
    """
    Selects representative "no snow" pixels by clustering and distance to cluster centroids.
    Saves the output as a raster.

    Parameters
    ----------
    bands_data : numpy.ndarray
        3D array (bands, height, width) containing spectral data for each band.
    valid_mask : numpy.ndarray
        2D mask of valid pixels for selection.
    k : int, optional
        Number of clusters for K-means, by default 5.
    n_closest : int, optional
        Number of closest pixels to each centroid to select, by default 5.

    Returns
    -------
    representative_pixels_mask : numpy.ndarray
        2D mask with representative pixels marked as 1.
    """
    # Extract "valid" pixels for clustering
    valid_pixels = bands_data[valid_mask, :]  # Shape (pixels, bands)

    # Normalize the valid pixels
    scaler = StandardScaler()
    normalized_pixels = scaler.fit_transform(valid_pixels)

    # find optimal K
    if k == 'auto':
        k = find_optimal_k(normalized_pixels, max_k=10, method="elbow")
    if n_closest == 'auto':
        n_closest = int(sample_count / k)

    # Perform K-means clustering on "no snow" pixels
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(normalized_pixels)

    # Get cluster centroids and labels
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Initialize an empty mask for representative pixels
    representative_pixels_mask = np.zeros(valid_mask.shape, dtype='uint8')

    # Find the n_closest pixels to each centroid
    for cluster_idx in range(k):
        # Select pixels in the current cluster
        cluster_indices = np.where(labels == cluster_idx)[0]
        cluster_pixels = normalized_pixels[cluster_indices]

        # Compute distances to the centroid for these pixels
        distances = distance.cdist(cluster_pixels, [centroids[cluster_idx]], 'euclidean').flatten()

        # Get the indices of the n_closest pixels in the cluster
        closest_indices = np.argsort(distances)[:n_closest]

        # Map the closest indices back to the original image coordinates
        original_indices = np.argwhere(valid_mask)[cluster_indices]
        selected_pixels = original_indices[closest_indices]

        # Set these pixels in the representative mask
        representative_pixels_mask[selected_pixels] = 1

    return representative_pixels_mask


def find_optimal_k(data, max_k=10, method="elbow", random_state=42):
    """
    Find the optimal number of clusters using the Elbow or Silhouette method.

    Parameters:
    - data (array-like): The dataset to cluster.
    - max_k (int): The maximum number of clusters to evaluate.
    - method (str): "elbow" for WCSS-based elbow method or "silhouette" for silhouette score.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - int: The optimal number of clusters.
    """
    wcss = []  # Within-Cluster Sum of Squares
    silhouette_scores = []  # Silhouette Scores
    k_values = range(2, max_k + 1)  # Start from 2 clusters for silhouette

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    if method == "elbow":
        # Calculate second derivative to find the "elbow"
        wcss_diff = np.diff(wcss)
        wcss_diff2 = np.diff(wcss_diff)
        optimal_k = k_values[np.argmin(wcss_diff2) + 1]  # Offset for the diff
    elif method == "silhouette":
        # Choose k with the highest silhouette score
        optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        raise ValueError("Invalid method. Choose 'elbow' or 'silhouette'.")

    return optimal_k