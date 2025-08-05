import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from dbscan_clustering import dbscan_deconet, merge_clusters
from bs_mode_control import (
    calculate_thinning_radius,
    compute_sinr,
    compute_energy_per_info_bit,
    apply_mode_control_multi
)


def plot_clusters(users, clusters, BS_positions=None, thinning_radii=None, bs_states=None, title="D-DeCoNet (DBSCAN-Based) with Thinning"):
    plt.figure(figsize=(10, 10))
    clustered_points = set()
    if clusters:
        for cluster in clusters:
            clustered_points.update(cluster)
    unclustered = [idx for idx in range(len(users)) if idx not in clustered_points]
    plt.scatter(users[unclustered, 0], users[unclustered, 1], c='lightgray', s=8, label='Unclustered')

    colormap = plt.cm.get_cmap('tab10', len(clusters))
    for index, cluster in enumerate(clusters):
        cluster_points = users[cluster]
        color = colormap(index)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=color, label=f'Major Cluster {index + 1}')
        center = cluster_points.mean(axis=0)
        plt.plot(center[0], center[1], marker='*', markersize=12, color='black', label='_nolegend_')
        if thinning_radii is not None:
            circle = plt.Circle(center, thinning_radii[index], color=color, linestyle='--', alpha=0.2, fill=False)
            plt.gca().add_patch(circle)

    if BS_positions is not None and isinstance(bs_states, list):
        for i, bs in enumerate(BS_positions):
            state = bs_states[i] if bs_states else "AWAKE"
            color = 'green' if state == "AWAKE" else 'red'
            label = 'AWAKE BS' if state == "AWAKE" else 'SLEEP BS'
            plt.scatter(bs[0], bs[1], marker='x', c=color, s=60, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "_nolegend_")

    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main(K=None, visualize=True):
    # Generate network
    BS_positions, users_all, users_low, users_high = generate_network(K=K)

    print("\n=== Network Information ===")
    print(f"Total Users: {len(users_all)} | Base Stations: {len(BS_positions)}")

    # DBSCAN clustering
    eps = 50
    min_pts = 10
    clusters, labels = dbscan_deconet(users_all, eps=eps, min_pts=min_pts)

    print("\n=== Clustering Results ===")
    print(f"Initial clusters detected: {len(clusters)}")
    merged_clusters = merge_clusters(users_all, clusters, merge_radius=150, min_cluster_size=50)
    print(f"After merging: {len(merged_clusters)} major clusters")

    # Thinning radius calculation (per-cluster)
    thinning_radii = calculate_thinning_radius(
        merged_clusters, users_all, BS_positions, algo_type="D-DeCoNet"
    )

    # Mode control
    bs_states = apply_mode_control_multi(
        merged_clusters, users_all, BS_positions, thinning_radii
    )

    # SINR + eta_I
    sinr_dict = compute_sinr(users_all, bs_states, BS_positions)
    eta_i_bit = compute_energy_per_info_bit(sinr_dict)
    eta_i_Mbit = eta_i_bit * 1e6

    num_awake = bs_states.count("AWAKE")
    num_sleep = bs_states.count("SLEEP")

    print("\n=== Base Station States ===")
    print(f"AWAKE: {num_awake} | SLEEP: {num_sleep} | eta_I: {eta_i_Mbit:.4f} J/Mbit")

    if visualize:
        plot_clusters(users_all, merged_clusters, BS_positions, thinning_radii, bs_states)

    return {
        "K": K,
        "total_users": len(users_all),
        "awake_BS": num_awake,
        "sleep_BS": num_sleep,
        "total_BS": len(BS_positions),
        "eta_i": eta_i_Mbit
    }


if __name__ == "__main__":
    results = []
    for K in range(2000, 32000, 2000):
        print(f"\n\n=== Running D-DeCoNet with K = {K} ===")
        res = main(K=K, visualize=False)
        results.append(res)

    Ks = [r["K"] for r in results]
    etas = [r["eta_i"] for r in results]

    plt.figure()
    plt.plot(Ks, etas, marker='o', label="D-DeCoNet")
    plt.xlabel("K (users in high-density areas)")
    plt.ylabel("eta_I (J/Mbit)")
    plt.title("Energy per Information Bit vs K")
    plt.grid(True)
    plt.legend()
    plt.show()

