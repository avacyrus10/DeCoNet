import numpy as np
import matplotlib.pyplot as plt
import random

from generate_network import generate_network
from optics_clustering import optics_deconet, merge_clusters

from bs_mode_control import calculate_thinning_radius, apply_mode_control


def plot_clusters(users, clusters, BS_positions=None, bs_states=None, title="O-DeCoNet (OPTICS-Based) Clustering"):
    """
    Plot merged clusters, unclustered users, cluster centers, and base stations.
    """
    plt.figure(figsize=(10, 10))

    # Identify unclustered users
    clustered_points = set()
    if clusters:
        for cluster in clusters:
            clustered_points.update(cluster)
    unclustered = [idx for idx in range(len(users)) if idx not in clustered_points]

    # Plot unclustered users
    plt.scatter(users[unclustered, 0], users[unclustered, 1],
                c='lightgray', s=8, label='Unclustered')

    # Assign colors for clusters
    colormap = plt.cm.get_cmap('tab10', len(clusters))
    for index, cluster in enumerate(clusters):
        cluster_points = users[cluster]
        color = colormap(index)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=30, color=color, label=f'Major Cluster {index + 1}')

        # Plot cluster center as star
        cluster_coords = users[cluster]
        center = cluster_coords.mean(axis=0)
        plt.scatter(center[0], center[1], marker='*', c='black', s=200)

    # Plot base stations with AWAKE/SLEEP state
    if BS_positions is not None and bs_states is not None:
        BS_positions = np.array(BS_positions)
        awake_mask = np.array(bs_states) == "AWAKE"
        sleep_mask = ~awake_mask

        plt.scatter(BS_positions[awake_mask, 0], BS_positions[awake_mask, 1],
                    marker='x', c='green', s=60, label='AWAKE BS')
        plt.scatter(BS_positions[sleep_mask, 0], BS_positions[sleep_mask, 1],
                    marker='x', c='red', s=60, label='SLEEP BS')

    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_reachability(reachability):
    """
    Plot the reachability distances for OPTICS ordering.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(reachability, color='darkblue')
    plt.title("Reachability Distance (RD) Plot - O-DeCoNet")
    plt.xlabel("Order of Users (S_ORDER)")
    plt.ylabel("Reachability Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main simulation for O-DeCoNet using OPTICS-based clustering.
    """
    # Generate network
    BS_positions, users_all, _, _ = generate_network()

    print("\n=== Network Information ===")
    print(f"Total Users: {len(users_all)}")
    print(f"Base Stations: {len(BS_positions)}")

    # Run OPTICS-based clustering
    min_pts = 10
    xi = 0.05
    clusters, reachability, order = optics_deconet(users_all, min_pts=min_pts, xi=xi, distance_threshold=50)

    print("\n=== Clustering Results ===")
    print(f"Initial clusters detected: {len(clusters)}")

    # Merge clusters to form major groups
    merge_radius = 150
    min_cluster_size = 50
    merged_clusters = merge_clusters(users_all, clusters,
                                     merge_radius=merge_radius,
                                     min_cluster_size=min_cluster_size)

    print(f"After merging: {len(merged_clusters)} major clusters")
    for index, cluster in enumerate(merged_clusters):
        print(f"Major Cluster {index + 1} size: {len(cluster)}")

    # Calculate thinning radius and apply mode control
    thinning_radii = calculate_thinning_radius(merged_clusters, users_all, BS_positions,
                                               algo_type="O-DeCoNet", reachability=reachability)

    bs_states = apply_mode_control(merged_clusters, users_all, BS_positions, thinning_radii)

    asleep_count = bs_states.count("SLEEP")
    awake_count = bs_states.count("AWAKE")
    print(f"\nBase Station States: {awake_count} awake, {asleep_count} asleep")

    # Visualization
    plot_clusters(users_all, merged_clusters, BS_positions, bs_states)
    plot_reachability(reachability)


if __name__ == "__main__":
    main()

