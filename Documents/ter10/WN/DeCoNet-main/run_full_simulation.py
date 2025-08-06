import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from generate_network import generate_network
from bs_mode_control import calculate_thinning_radius, apply_mode_control, compute_power_per_area
from optics_clustering import optics_deconet, merge_clusters as optics_merge
from dbscan_clustering import dbscan_deconet, merge_clusters as dbscan_merge

def simulate_for_k(K_lambda, algo_type):
    """
    Simulates a single run for a given user density and algorithm.
    """
    BS_positions, users_all, _, _ = generate_network(lambda_u_high=K_lambda,
                                                     lambda_u_low=0,
                                                     num_high_density_areas=1)
    bs_states = []

    if algo_type == "AA":
        bs_states = ["AWAKE"] * len(BS_positions)
    elif algo_type == "ACEnet":
        dist_matrix = distance_matrix(users_all, BS_positions)
        nearest_BS_indices = np.argmin(dist_matrix, axis=1)
        awake_bs_indices = np.unique(nearest_BS_indices)
        bs_states = ["SLEEP"] * len(BS_positions)
        for idx in awake_bs_indices:
            bs_states[idx] = "AWAKE"
    elif algo_type == "O-DeCoNet":
        clusters, reachability, _ = optics_deconet(users_all)
        merged_clusters = optics_merge(users_all, clusters)
        thinning_radii = calculate_thinning_radius(merged_clusters, users_all, BS_positions, algo_type="O-DeCoNet", reachability=reachability)
        bs_states = apply_mode_control(merged_clusters, users_all, BS_positions, thinning_radii)
    elif algo_type == "D-DeCoNet":
        clusters, _ = dbscan_deconet(users_all)
        merged_clusters = dbscan_merge(users_all, clusters)
        thinning_radii = calculate_thinning_radius(merged_clusters, users_all, BS_positions, algo_type="D-DeCoNet")
        bs_states = apply_mode_control(merged_clusters, users_all, BS_positions, thinning_radii)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")

    eta_A = compute_power_per_area(bs_states, BS_positions)
    return eta_A

if __name__ == "__main__":
    K_values_plot = range(2, 32, 2)
    lambda_u_high_values = [k * 1000 for k in K_values_plot]
    num_runs = 50 
    
    aa_all_runs = [[] for _ in K_values_plot]
    acenet_all_runs = [[] for _ in K_values_plot]
    o_deconet_all_runs = [[] for _ in K_values_plot]
    d_deconet_all_runs = [[] for _ in K_values_plot]

    for run in range(num_runs):
        print(f"--- Running simulation {run + 1}/{num_runs} ---")
        for i, K_lambda in enumerate(lambda_u_high_values):
            print(f"Simulating for K = {K_lambda/1000}")
            o_deconet_all_runs[i].append(simulate_for_k(K_lambda, "O-DeCoNet"))
            d_deconet_all_runs[i].append(simulate_for_k(K_lambda, "D-DeCoNet"))
            acenet_all_runs[i].append(simulate_for_k(K_lambda, "ACEnet"))
            aa_all_runs[i].append(simulate_for_k(K_lambda, "AA"))
    
    aa_etas = [np.mean(runs) for runs in aa_all_runs]
    acenet_etas = [np.mean(runs) for runs in acenet_all_runs]
    o_deconet_etas = [np.mean(runs) for runs in o_deconet_all_runs]
    d_deconet_etas = [np.mean(runs) for runs in d_deconet_all_runs]
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(K_values_plot, o_deconet_etas, marker='s', label='O-DeCoNet')
    plt.plot(K_values_plot, d_deconet_etas, marker='o', label='D-DeCoNet')
    plt.plot(K_values_plot, acenet_etas, marker='^', label='ACEnet')
    plt.plot(K_values_plot, aa_etas, marker='x', label='AA')

    plt.xlabel('# of users in high user-density area (K)')
    plt.ylabel('Power per area unit ($W/km^2$)')
    plt.title('Energy Efficiency ($\eta_A$) vs. High User-Density (Averaged)')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0, 4000)
    plt.xlim(0, 32)
    plt.show()
