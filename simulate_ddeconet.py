import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from dbscan_clustering import dbscan_deconet, merge_clusters


def plot_clusters(users, clusters, BS_positions=None, title="D-DeCoNet (DBSCAN-Based) Merged Clustering"):
    """
    Plot the merged clusters along with unclustered users and Base Station (BS) positions.
    """
    plt.figure(figsize=(10, 10))

    # Identify unclustered users
    clustered_points = set()
    if clusters:
        for cluster in clusters:
            clustered_points.update(cluster)

    unclustered = []
    for idx in range(len(users)):
        if idx not in clustered_points:
            unclustered.append(idx)

    # Plot unclustered users
    plt.scatter(users[unclustered, 0], users[unclustered, 1], c='lightgray', s=8, label='Unclustered')

    # Plot each cluster with a unique color
    colors = plt.cm.get_cmap('tab10', len(clusters))
    for idx, cluster in enumerate(clusters):
        cluster_points = users[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors(idx), label=f'Major Cluster {idx + 1}')

    # Plot Base Stations
    if BS_positions is not None:
        plt.scatter(BS_positions[:, 0], BS_positions[:, 1], marker='x', c='black', s=50, label='Base Stations')

    # Chart settings
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    """
    Simulate D-DeCoNet clustering using DBSCAN and visualize results.
    """
    # Step 1: Generate the network using the existing function
    BS_positions, users_all, users_low, users_high = generate_network()

    print(f"Total Users: {len(users_all)}, Low-density: {len(users_low)}, High-density: {len(users_high)}")
    print(f"Base Stations: {len(BS_positions)}")

    # Step 2: Apply DBSCAN clustering
    eps = 50  # Neighborhood radius (meters)
    min_pts = 10  # Minimum points to form a core cluster
    clusters, labels = dbscan_deconet(users_all, eps=eps, min_pts=min_pts)
    print(f"Initial clusters detected by DBSCAN: {len(clusters)}")

    # Step 3: Merge clusters into major groups
    merge_radius = 150  # Distance threshold for merging clusters
    min_cluster_size = 50  # Ignore small clusters
    merged_clusters = merge_clusters(users_all, clusters, merge_radius=merge_radius, min_cluster_size=min_cluster_size)
    print(f"After merging: {len(merged_clusters)} major clusters")

    for idx, cl in enumerate(merged_clusters):
        print(f"Major Cluster {idx + 1} size: {len(cl)}")

    # Step 4: Visualize the final result
    plot_clusters(users_all, merged_clusters, BS_positions)


if __name__ == "__main__":
    main()

