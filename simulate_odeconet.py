import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from optics_clustering import optics_deconet, merge_clusters


def plot_clusters(users, clusters, BS_positions=None, title="O-DeCoNet (OPTICS-Based) Clustering"):
    """
    Plot major clusters after merging and unclustered points.

    Args:
        users (ndarray): Coordinates of all users
        clusters (list): List of clusters (list of indices)
        BS_positions (ndarray): Base station coordinates
        title (str): Plot title
    """
    plt.figure(figsize=(10, 10))

    # Identify unclustered users
    clustered_points = set()
    if clusters:
        for cluster in clusters:
            clustered_points.update(cluster)
    unclustered = [idx for idx in range(len(users)) if idx not in clustered_points]

    # Plot unclustered users
    plt.scatter(users[unclustered, 0], users[unclustered, 1], c='lightgray', s=8, label='Unclustered')

    # Plot clusters with different colors
    color_map = plt.cm.get_cmap('tab10', len(clusters))
    for idx, cluster in enumerate(clusters):
        cluster_coords = users[cluster]
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=30, color=color_map(idx), label=f'Major Cluster {idx+1}')

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
    """
    Plot the reachability distances for OPTICS ordering.

    Args:
        reachability (list): Reachability distances in the OPTICS order
    """
    plt.figure(figsize=(10, 4))
    plt.plot(reachability, color='darkblue')
    plt.title("Reachability Distance (RD) Plot - O-DeCoNet")
    plt.xlabel("Order of Users (S_ORDER)")
    plt.ylabel("Reachability Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# Main Simulation Logic
# =========================
if __name__ == "__main__":
    # Generate the network
    BS_positions, users_all, _, _ = generate_network()

    print(f"Total Users: {len(users_all)}")
    print(f"Base Stations: {len(BS_positions)}")

    # Run OPTICS-based clustering
    min_pts = 10
    xi = 0.05
    clusters, reachability, order = optics_deconet(users_all, min_pts=min_pts, xi=xi, distance_threshold=50)

    print(f"Initial clusters found: {len(clusters)}")

    # Merge clusters to form major groups
    merged_clusters = merge_clusters(users_all, clusters, merge_radius=150, min_cluster_size=50)
    print(f"After merging: {len(merged_clusters)} major clusters")
    for idx, cluster in enumerate(merged_clusters):
        print(f"Major Cluster {idx+1} size: {len(cluster)}")

    # Visualization
    plot_clusters(users_all, merged_clusters, BS_positions)
    plot_reachability(reachability)

