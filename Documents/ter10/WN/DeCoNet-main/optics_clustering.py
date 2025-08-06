import numpy as np
from sklearn.cluster import OPTICS

def optics_deconet(users, min_pts=10, xi=0.05):
    """
    Implements O-DeCoNet OPTICS-based clustering using scikit-learn.
    """
    optics = OPTICS(min_samples=min_pts, xi=xi, min_cluster_size=min_pts).fit(users)
    
    labels = optics.labels_
    
    clusters = []
    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        cluster_indices = np.where(labels == cluster_id)[0].tolist()
        clusters.append(cluster_indices)
        
    reachability = optics.reachability_
    
    return clusters, reachability, optics.ordering_

def merge_clusters(users, clusters, merge_radius=150, min_cluster_size=1):
    """
    Merge clusters that are close together into major clusters.
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

    for i, cluster_i in enumerate(large_clusters):
        if i in used:
            continue

        merged_set = set(cluster_i)
        for j in range(i + 1, len(large_clusters)):
            if np.linalg.norm(centroids[i] - centroids[j]) < merge_radius:
                merged_set.update(large_clusters[j])
                used.add(j)

        merged_clusters.append(list(merged_set))

    return merged_clusters
