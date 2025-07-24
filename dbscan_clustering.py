import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_deconet(users, eps=30, min_pts=5):
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

