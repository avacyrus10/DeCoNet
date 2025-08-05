import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from optics_clustering import optics_deconet, merge_clusters
from bs_mode_control import (
    calculate_thinning_radius,
    compute_sinr,
    compute_energy_per_info_bit,
    apply_mode_control_multi
)


def plot_clusters(users, clusters, BS_positions=None, bs_states=None, title="O-DeCoNet (OPTICS-Based) Clustering"):
    plt.figure(figsize=(10, 10))

    clustered_points = set()
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
        plt.scatter(center[0], center[1], marker='*', c='black', s=200, label='_nolegend_')

    if BS_positions is not None and bs_states is not None:
        BS_positions = np.array(BS_positions)
        awake_mask = np.array(bs_states) == "AWAKE"
        sleep_mask = ~awake_mask
        plt.scatter(BS_positions[awake_mask, 0], BS_positions[awake_mask, 1], marker='x', c='green', s=60, label='AWAKE BS')
        plt.scatter(BS_positions[sleep_mask, 0], BS_positions[sleep_mask, 1], marker='x', c='red', s=60, label='SLEEP BS')

    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_reachability(reachability):
    plt.figure(figsize=(10, 4))
    plt.plot(reachability, color='darkblue')
    plt.title("Reachability Distance (RD) Plot - O-DeCoNet")
    plt.xlabel("Order of Users (S_ORDER)")
    plt.ylabel("Reachability Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(K=None, visualize=True):
    BS_positions, users_all, _, _ = generate_network(K=K)

    print("\n=== Network Information ===")
    print(f"Total Users: {len(users_all)} | Base Stations: {len(BS_positions)}")

    # OPTICS clustering
    min_pts = 10
    xi = 0.05
    clusters, reachability, order = optics_deconet(
        users_all, min_pts=min_pts, xi=xi, distance_threshold=50
    )

    print("\n=== Clustering Results ===")
    print(f"Initial clusters detected: {len(clusters)}")
    merged_clusters = merge_clusters(users_all, clusters, merge_radius=150, min_cluster_size=50)
    print(f"After merging: {len(merged_clusters)} major clusters")

    # Thinning radius calculation
    thinning_radii = calculate_thinning_radius(
        merged_clusters, users_all, BS_positions, algo_type="O-DeCoNet", reachability=reachability
    )

    # Use the corrected mode control function.
    bs_states = apply_mode_control_multi(
        merged_clusters, users_all, BS_positions, thinning_radii
    )

    # SINR + ηᵢ
    sinr_dict = compute_sinr(users_all, bs_states, BS_positions)
    eta_i_bit = compute_energy_per_info_bit(sinr_dict)  # J/bit
    eta_i_Mbit = eta_i_bit * 1e6

    awake = bs_states.count("AWAKE")
    sleep = bs_states.count("SLEEP")

    print(f"\nBase Station States: {awake} AWAKE, {sleep} SLEEP | ηᵢ: {eta_i_Mbit:.4f} J/Mbit")

    if visualize:
        plot_clusters(users_all, merged_clusters, BS_positions, bs_states)
        plot_reachability(reachability)

    return {
        "K": K,
        "total_users": len(users_all),
        "awake_BS": awake,
        "sleep_BS": sleep,
        "total_BS": len(BS_positions),
        "eta_i": eta_i_Mbit
    }


if __name__ == "__main__":
    results = []
    # K range should be adjusted to match the paper's graph.
    # The paper shows K from 2 to 30, which corresponds to user densities.
    # Your K is in thousands. Let's assume K in your code is the total users.
    for K_val in range(2000, 32000, 2000):
        print(f"\n\n=== Running O-DeCoNet with K = {K_val} ===")
        res = main(K=K_val, visualize=False)
        results.append(res)

    Ks = [r["K"] for r in results]
    etas = [r["eta_i"] for r in results]

    plt.figure()
    plt.plot(Ks, etas, marker='o', label="O-DeCoNet")
    plt.xlabel("K (users in high-density areas)")
    plt.ylabel("ηᵢ (J/Mbit)")
    plt.title("Energy per Information Bit vs K (O-DeCoNet)")
    plt.grid(True)
    plt.legend()
    plt.show()

