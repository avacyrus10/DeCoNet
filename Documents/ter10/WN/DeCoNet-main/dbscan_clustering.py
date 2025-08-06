import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_deconet(users, eps=50, min_pts=10):
    """
    Implements D-DeCoNet clustering using DBSCAN from scikit-learn.
    
    Args:
        users (ndarray): shape (N, 2) array of user coordinates
        eps (float): ε - maximum neighborhood distance
        min_pts (int): MinPts - minimum number of users to form a core

    Returns:
        clusters (list of list): List of clusters (each is list of indices)
        labels (ndarray): label for each user (-1 means noise)
    """
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(users)
    labels = db.labels_

    clusters = []
    unique_labels = set(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        
        cluster_indices = np.where(labels == cluster_id)[0].tolist()
        clusters.append(cluster_indices)

    return clusters, labels

def merge_clusters(users, clusters, merge_radius=150, min_cluster_size=1):
    """
    Merge clusters that are close to each other into major clusters.

    Args:
        users (ndarray): All user positions
        clusters (list): List of initial clusters
        merge_radius (float): Maximum distance between cluster centroids to merge
        min_cluster_size (int): Ignore clusters smaller than this size

    Returns:
        merged_clusters (list of list): List of merged major clusters
    """
    large_clusters = []
    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            large_clusters.append(cluster)
    
    centroids = []
    if not large_clusters:
        return []
    for cluster in large_clusters:
        centroid = users[cluster].mean(axis=0)
        centroids.append(centroid)

    merged_clusters = []
    used = set()

    for i, cluster in enumerate(large_clusters):
        if i in used:
            continue
        
        merged = set(cluster)
        for j in range(i + 1, len(large_clusters)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < merge_radius:
                merged.update(large_clusters[j])
                used.add(j)

        merged_clusters.append(list(merged))

    return merged_clusters
