import numpy as np
from sklearn.metrics import pairwise_distances


def optics_deconet(users, min_pts=10, xi=0.05, distance_threshold=None):
    """
    Implements O-DeCoNet OPTICS-based clustering.

    Args:
        users (ndarray): Array of user coordinates (N, 2)
        min_pts (int): Minimum number of points to define a core
        xi (float): Steepness parameter for reachability distance (RD) valley detection
        distance_threshold (float): Optional fallback threshold for clustering

    Returns:
        clusters (list): List of clusters (each is a list of indices)
        reachability (list): Reachability distances for each user in order
        S_ORDER (list): Ordering of users as processed
    """
    N = users.shape[0]

    # Compute pairwise distances
    distance_matrix = pairwise_distances(users)

    # Compute core distances (CD)
    core_distances = np.full(N, np.inf)
    for i in range(N):
        sorted_distances = np.sort(distance_matrix[i])
        if len(sorted_distances) > min_pts:
            core_distances[i] = sorted_distances[min_pts]

    # Initialize reachability distances (RD) and seeds
    reachability_distances = np.full(N, np.inf)
    processing_order = []
    seed_set = set(range(N))

    # OPTICS ordering process
    while seed_set:
        if np.isfinite(reachability_distances).any():
            current_point = min(seed_set, key=lambda x: reachability_distances[x])
        else:
            current_point = np.random.choice(list(seed_set))

        processing_order.append(current_point)
        seed_set.remove(current_point)

        # Update RD for remaining points
        for neighbor in seed_set:
            distance_to_neighbor = distance_matrix[current_point][neighbor]
            updated_rd = max(core_distances[current_point], distance_to_neighbor)
            if updated_rd < reachability_distances[neighbor]:
                reachability_distances[neighbor] = updated_rd

    # Cluster extraction using RD valleys
    clusters = []
    i = 0
    while i < len(processing_order) - 1:
        j = i
        while j < len(processing_order) - 1 and reachability_distances[processing_order[j]] * (1 - xi) >= reachability_distances[processing_order[j + 1]]:
            j += 1

        k = j
        while k < len(processing_order) - 1 and reachability_distances[processing_order[k]] <= reachability_distances[processing_order[k + 1]] * (1 - xi):
            k += 1

        if (k - j) >= min_pts:
            cluster = processing_order[j + 1:k + 1]
            clusters.append(cluster)
            i = k + 1
        else:
            i += 1

    # Fallback clustering if no clusters found
    if len(clusters) == 0 and distance_threshold is not None:
        visited = np.zeros(N, dtype=bool)
        for idx in range(N):
            if visited[idx]:
                continue
            neighbors = np.where(distance_matrix[idx] <= distance_threshold)[0]
            if len(neighbors) >= min_pts:
                clusters.append(list(neighbors))
                visited[neighbors] = True

    reachability = [reachability_distances[i] for i in processing_order]
    return clusters, reachability, processing_order


def merge_clusters(users, clusters, merge_radius=150, min_cluster_size=50):
    """
    Merge clusters that are close together into major clusters.

    Args:
        users (ndarray): All user positions
        clusters (list): List of initial clusters
        merge_radius (float): Maximum distance between cluster centroids to merge
        min_cluster_size (int): Ignore clusters smaller than this size

    Returns:
        merged_clusters (list of list): List of merged major clusters
    """
    # Filter clusters based on minimum size
    large_clusters = []
    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            large_clusters.append(cluster)

    # Compute centroids for large clusters
    centroids = []
    for cluster in large_clusters:
        centroid = users[cluster].mean(axis=0)
        centroids.append(centroid)

    merged_clusters = []
    used = set()

    # Merge clusters that are close together
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

