import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, distance_matrix


def max_pairwise_distance(points: np.ndarray) -> float:
    """
    Compute the maximum pairwise distance in a set of points
    """
    n = len(points)

    if n <= 1:
        return 0.0

    if n == 2:
        diff = points[0] - points[1]
        single_distance = np.linalg.norm(diff)
        return float(single_distance)

    # Use the convex hull to reduce the number of pair checks
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    dmat = distance_matrix(hull_points, hull_points)
    max_distance = np.max(dmat)

    return float(max_distance)


def dbscan_deconet(users, eps=50, min_pts=10, merge_radius=150, min_cluster_size=50):
    """
    Full D-DeCoNet: DBSCAN clustering, merge, and compute r_c for each cluster.
    r_c = max distance between any two users in the cluster / 2
    """
    db = DBSCAN(
        eps=eps,
        min_samples=min_pts,
        metric='euclidean',
        algorithm='ball_tree'
    )
    db.fit(users)

    labels = db.labels_

    # Initial clusters (skip noise label == -1)
    clusters = []
    unique_labels = set(labels)
    for cid in unique_labels:
        if cid == -1:
            continue
        members = np.where(labels == cid)[0].tolist()
        clusters.append(members)

    # Merge close clusters
    merged_clusters = merge_clusters(
        users=users,
        clusters=clusters,
        merge_radius=merge_radius,
        min_cluster_size=min_cluster_size
    )

    # Compute r_c for each merged cluster
    rc_list = []
    for cluster in merged_clusters:
        # If the merged cluster is too small to define a radius, use 0.0
        if len(cluster) < 2:
            rc_list.append(0.0)
            continue

        max_dist = max_pairwise_distance(users[cluster])
        rc = max_dist / 2.0
        rc_list.append(float(rc))

    return merged_clusters, rc_list


def merge_clusters(users, clusters, merge_radius=150, min_cluster_size=50):
    """
    Merge clusters that are close together based on centroid distance.
    """
    # Keep only clusters meeting the minimum size
    large_clusters = []
    for c in clusters:
        if len(c) >= min_cluster_size:
            large_clusters.append(c)

    # If nothing qualifies, return empty list
    if len(large_clusters) == 0:
        return []

    # Compute centroids for each large cluster
    centroids = []
    for c in large_clusters:
        centroid = users[c].mean(axis=0)
        centroids.append(centroid)

    merged_clusters = []
    used = set()

    for i, cluster_i in enumerate(large_clusters):
        if i in used:
            continue

        current_merged = set(cluster_i)

        for j in range(i + 1, len(large_clusters)):
            if j in used:
                continue

            # Distance between centroids
            diff = centroids[i] - centroids[j]
            centroid_distance = np.linalg.norm(diff)

            if centroid_distance < float(merge_radius):
                current_merged.update(large_clusters[j])
                used.add(j)

        merged_clusters.append(list(current_merged))

    return merged_clusters

