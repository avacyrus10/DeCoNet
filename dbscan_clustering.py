import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_deconet(users, eps=50, min_pts=10):
    """
    Implements D-DeCoNet clustering using DBSCAN.

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
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise
        cluster_indices = np.where(labels == cluster_id)[0].tolist()
        clusters.append(cluster_indices)

    return clusters, labels


def merge_clusters(users, clusters, merge_radius=150, min_cluster_size=50):
    """
    Merge clusters that are close to each other into major clusters.

    Args:
        users (ndarray): All user positions
        clusters (list): List of initial clusters (lists of indices)
        merge_radius (float): Max distance between centroids to merge
        min_cluster_size (int): Minimum cluster size to retain

    Returns:
        merged_clusters (list of list): Merged list of major clusters
    """
    # Keep only large clusters
    large_clusters = [c for c in clusters if len(c) >= min_cluster_size]
    centroids = [users[c].mean(axis=0) for c in large_clusters]

    merged_clusters = []
    used = set()

    for i, cluster in enumerate(large_clusters):
        if i in used:
            continue

        merged = set(cluster)
        for j in range(i + 1, len(large_clusters)):
            if j in used:
                continue
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < merge_radius:
                merged.update(large_clusters[j])
                used.add(j)

        merged_clusters.append(list(merged))

    return merged_clusters

