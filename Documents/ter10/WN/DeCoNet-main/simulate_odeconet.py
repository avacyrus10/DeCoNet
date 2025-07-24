import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# =========================
# OPTICS-based Clustering
# =========================
def optics_deconet(users, min_pts=5, xi=0.1, distance_threshold=None):
    """
    Implements O-DeCoNet OPTICS-based clustering.
    
    Args:
        users: ndarray (N,2) user coordinates
        min_pts: minimum number of points to define a core
        xi: steepness parameter for RD valley detection
        distance_threshold: optional fallback threshold for clustering

    Returns:
        clusters: list of clusters (list of indices)
        reachability: RD list for users in order
        S_ORDER: ordering of users
    """
    N = users.shape[0]
    D = pairwise_distances(users)

    # Core Distances (CD)
    CD = np.full(N, np.inf)
    for i in range(N):
        sorted_dists = np.sort(D[i])
        if len(sorted_dists) > min_pts:
            CD[i] = sorted_dists[min_pts]

    # Initialize Reachability (RD)
    RD = np.full(N, np.inf)
    S_ORDER = []
    SSEED = set(range(N))

    while SSEED:
        # Pick point with smallest RD (or random if all inf)
        i = min(SSEED, key=lambda x: RD[x]) if np.isfinite(RD).any() else np.random.choice(list(SSEED))
        S_ORDER.append(i)
        SSEED.remove(i)

        # Update RD for remaining points
        for k in SSEED:
            dist = D[i][k]
            TV = max(CD[i], dist)
            if TV < RD[k]:
                RD[k] = TV

    # =========================
    # Cluster extraction using RD valleys
    # =========================
    clusters = []
    i = 0
    while i < len(S_ORDER) - 1:
        j = i
        while j < len(S_ORDER) - 1 and RD[S_ORDER[j]] * (1 - xi) >= RD[S_ORDER[j + 1]]:
            j += 1
        k = j
        while k < len(S_ORDER) - 1 and RD[S_ORDER[k]] <= RD[S_ORDER[k + 1]] * (1 - xi):
            k += 1

        if k - j >= min_pts:
            cluster = S_ORDER[j + 1:k + 1]
            clusters.append(cluster)
            i = k + 1
        else:
            i += 1

    # =========================
    # Fallback: if no clusters found
    # =========================
    if len(clusters) == 0 and distance_threshold is not None:
        visited = np.zeros(N, dtype=bool)
        for idx in range(N):
            if visited[idx]:
                continue
            neighbors = np.where(D[idx] <= distance_threshold)[0]
            if len(neighbors) >= min_pts:
                clusters.append(list(neighbors))
                visited[neighbors] = True

    reachability = [RD[i] for i in S_ORDER]
    return clusters, reachability, S_ORDER


# =========================
# Visualization Functions
# =========================
def plot_clusters(users, clusters, BS_positions=None, title="O-DeCoNet Clustering", min_cluster_size=50, merge_radius=150):
    from sklearn.metrics import pairwise_distances
    plt.figure(figsize=(10, 10))

    # Filter and merge clusters
    large_clusters = [c for c in clusters if len(c) >= min_cluster_size]
    centroids = [users[c].mean(axis=0) for c in large_clusters]

    merged_clusters = []
    used = set()
    for i, c in enumerate(large_clusters):
        if i in used:
            continue
        merged = set(c)
        for j in range(i+1, len(large_clusters)):
            if np.linalg.norm(centroids[i] - centroids[j]) < merge_radius:
                merged.update(large_clusters[j])
                used.add(j)
        merged_clusters.append(list(merged))

    # Plot unclustered points
    clustered_points = set(np.concatenate(merged_clusters)) if merged_clusters else set()
    unclustered = [idx for idx in range(len(users)) if idx not in clustered_points]
    plt.scatter(users[unclustered, 0], users[unclustered, 1], c='lightgray', s=8, label='Unclustered')

    # Plot merged clusters
    colors = plt.cm.get_cmap('tab10', len(merged_clusters))
    for idx, cluster in enumerate(merged_clusters):
        coords = users[cluster]
        plt.scatter(coords[:, 0], coords[:, 1], s=30, color=colors(idx), label=f'Major Cluster {idx+1}')

    # Plot base stations
    if BS_positions is not None:
        plt.scatter(BS_positions[:, 0], BS_positions[:, 1], marker='x', c='black', s=50, label='Base Stations')

    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()



def plot_reachability(reachability):
    plt.figure(figsize=(10, 4))
    plt.plot(reachability, color='darkblue')
    plt.title("Reachability Distance (RD) Plot - O-DeCoNet")
    plt.xlabel("Order of users (S_ORDER)")
    plt.ylabel("Reachability Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# Main Testing
# =========================
if __name__ == "__main__":

    from generate_network import generate_network

    BS_positions, users_all, _, _ = generate_network()

    min_pts = 10
    xi = 0.05
    min_cluster_size = 50
    clusters, reachability, order = optics_deconet(users_all, min_pts=min_pts, xi=xi, distance_threshold=50)

    print(f"Total clusters found: {len(clusters)}")
    if clusters:
        for idx, cl in enumerate(clusters[:3]):
            print(f"Cluster {idx+1} size: {len(cl)}")

    plot_clusters(users_all, clusters, BS_positions, title="O-DeCoNet (OPTICS-Based) Clustering of Users")
    plot_reachability(reachability)

