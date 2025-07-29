import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from dbscan_clustering import dbscan_deconet, merge_clusters
from bs_mode_control import calculate_thinning_radius, apply_mode_control  # ✅ Use existing logic


def plot_clusters(users, clusters, BS_positions=None, thinning_radii=None, bs_states=None, title="D-DeCoNet (DBSCAN-Based) with Thinning"):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 10))

    # Identify unclustered users
    clustered_points = set()
    if clusters:
        for cluster in clusters:
            clustered_points.update(cluster)

    unclustered = [idx for idx in range(len(users)) if idx not in clustered_points]

    # Plot unclustered users in light gray
    plt.scatter(users[unclustered, 0], users[unclustered, 1],
                c='lightgray', s=8, label='Unclustered')

    # Assign colors for clusters
    colormap = plt.cm.get_cmap('tab10', len(clusters))
    for index, cluster in enumerate(clusters):
        cluster_points = users[cluster]
        color = colormap(index)

        # Plot users in cluster
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=30, color=color, label=f'Major Cluster {index + 1}')

        # Calculate and plot cluster center
        center = cluster_points.mean(axis=0)  # <-- NEW
        plt.plot(center[0], center[1], marker='*', markersize=12, color='black', label='_nolegend_')  # <-- NEW

        # Draw thinning circle if given
        if thinning_radii is not None:
            circle = plt.Circle(center, thinning_radii[index], color=color, linestyle='--', alpha=0.2, fill=False)
            plt.gca().add_patch(circle)

    # Plot Base Stations
    if BS_positions is not None and isinstance(bs_states, list):
        for i, bs in enumerate(BS_positions):
            state = bs_states[i] if bs_states else "AWAKE"
            color = 'green' if state == "AWAKE" else 'red'
            label = 'AWAKE BS' if state == "AWAKE" else 'SLEEP BS'
            plt.scatter(bs[0], bs[1], marker='x', c=color, s=60, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "_nolegend_")

    # Configure plot
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.axis("equal")
    plt.show()



def main():
    """
    Main simulation for D-DeCoNet using DBSCAN-based clustering with BS mode control.
    """
    # Generate network
    BS_positions, users_all, users_low, users_high = generate_network()

    print("\n=== Network Information ===")
    print(f"Total Users: {len(users_all)} | Base Stations: {len(BS_positions)}")

    # DBSCAN clustering
    eps = 50
    min_pts = 10
    clusters, labels = dbscan_deconet(users_all, eps=eps, min_pts=min_pts)

    print("\n=== Clustering Results ===")
    print(f"Initial clusters detected by DBSCAN: {len(clusters)}")

    # Merge clusters
    merged_clusters = merge_clusters(users_all, clusters, merge_radius=150, min_cluster_size=50)
    print(f"After merging: {len(merged_clusters)} major clusters")

    # Calculate thinning radii
    thinning_radii = calculate_thinning_radius(merged_clusters, users_all, BS_positions)

    # Apply mode control
    bs_states = apply_mode_control(merged_clusters, users_all, BS_positions, thinning_radii)

    print("\n=== Base Station States ===")
    print(f"AWAKE: {bs_states.count('AWAKE')} | SLEEP: {bs_states.count('SLEEP')}")

    # Visualization
    plot_clusters(
    users=users_all,
    clusters=merged_clusters,
    BS_positions=BS_positions,
    thinning_radii=thinning_radii,
    bs_states=bs_states
)



if __name__ == "__main__":
    main()
