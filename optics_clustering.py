import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.cluster import OPTICS


def _max_pairwise_distance(points: np.ndarray) -> float:
    n = len(points)

    if n <= 1:
        return 0.0

    if n == 2:
        return float(np.linalg.norm(points[0] - points[1]))

    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    dmat = distance_matrix(hull_pts, hull_pts)
    max_distance = np.max(dmat)

    return float(max_distance)


def optics_deconet(
    users,
    min_pts: int = 10,
    xi: float = 0.10,
    min_cluster_size: int = 50,
    merge_radius: float = 150.0
):
    """
    O-DeCoNet using sklearn OPTICS + xi clustering.

    Returns:
        merged_clusters: list[list[int]]  (indices into `users`)
        rc_list: list[float] radius per merged cluster
    """
    if users.shape[0] == 0:
        return [], []

    # sklearn OPTICS with 'xi' method reliably finds density valleys
    optics = OPTICS(
        min_samples=min_pts,
        xi=xi,
        min_cluster_size=min_cluster_size,
        cluster_method="xi",
        metric="euclidean",
        n_jobs=-1
    )
    optics.fit(users)

    labels = optics.labels_
    reach = optics.reachability_
    ordering = optics.ordering_  # Not currently used

    # Build raw clusters from labels
    raw_clusters = []
    unique_labels = set(labels)

    for cid in unique_labels:
        if cid == -1:
            continue
        member_indices = np.where(labels == cid)[0].tolist()
        raw_clusters.append(member_indices)

    # Optionally merge nearby clusters
    merged_clusters = merge_clusters(users, raw_clusters, merge_radius, min_cluster_size)

    # Compute r_c for each merged cluster
    rc_list = []
    for cl in merged_clusters:
        if not cl:
            rc_list.append(0.0)
            continue

        rvals = reach[np.array(cl)]
        rvals = rvals[np.isfinite(rvals)]

        if rvals.size > 0:
            mean_r = np.mean(rvals)
            if mean_r > 0:
                rc_list.append(float(mean_r))
                continue

        # If reachability is not usable, fall back to half of max pairwise distance
        fallback_radius = _max_pairwise_distance(users[cl]) / 2.0
        rc_list.append(fallback_radius)

    return merged_clusters, rc_list


def merge_clusters(users, clusters, merge_radius=150.0, min_cluster_size=50):
    # Keep only clusters that meet the minimum size
    large_clusters = []
    for c in clusters:
        if len(c) >= min_cluster_size:
            large_clusters.append(c)

    if not large_clusters:
        return []

    # Compute centroids for large clusters
    centroids = []
    for c in large_clusters:
        centroid = users[c].mean(axis=0)
        centroids.append(centroid)

    merged_clusters = []
    used = set()

    for i, ci in enumerate(large_clusters):
        if i in used:
            continue

        combined_cluster = set(ci)

        for j in range(i + 1, len(large_clusters)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < merge_radius:
                combined_cluster.update(large_clusters[j])
                used.add(j)

        merged_clusters.append(list(combined_cluster))

    return merged_clusters

