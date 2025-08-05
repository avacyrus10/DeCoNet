import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from scipy.spatial import distance_matrix
from bs_mode_control import compute_energy_per_info_bit, compute_sinr

def simulate_aa(K):
    """
    All base stations are active.
    """
    BS_positions, users_all, _, _ = generate_network(K=K)
    
    # All BSs are awake
    bs_states = ["AWAKE"] * len(BS_positions)
    
    # Compute SINR for all users
    sinr_dict = compute_sinr(users_all, bs_states, BS_positions)
    
    # Compute eta_I using the SINR results
    eta_i_bit = compute_energy_per_info_bit(sinr_dict, verbose=False)
    eta_i_Mbit = eta_i_bit * 1e6

    return {
        "K": K,
        "awake_BS": len(BS_positions),
        "sleep_BS": 0,
        "total_BS": len(BS_positions),
        "eta_i": eta_i_Mbit
    }

def simulate_acenet(K):
    """
    Each user connects to its nearest BS. BSs with no assigned users are put to sleep.
    """
    BS_positions, users_all, _, _ = generate_network(K=K)
    num_BS = len(BS_positions)
    
    # Compute distance matrix between users and BSs
    dist_matrix = distance_matrix(users_all, BS_positions)
    nearest_BS_indices = np.argmin(dist_matrix, axis=1)

    # Determine which BSs are awake
    awake_bs_indices = np.unique(nearest_BS_indices)
    bs_states = ["SLEEP"] * num_BS
    for idx in awake_bs_indices:
        bs_states[idx] = "AWAKE"

    # Compute SINR for all users
    sinr_dict = compute_sinr(users_all, bs_states, BS_positions)
    
    # Compute eta_I using the SINR results
    eta_i_bit = compute_energy_per_info_bit(sinr_dict, verbose=False)
    eta_i_Mbit = eta_i_bit * 1e6

    awake_BS = len(awake_bs_indices)
    sleep_BS = num_BS - awake_BS
    
    return {
        "K": K,
        "awake_BS": awake_BS,
        "sleep_BS": sleep_BS,
        "total_BS": num_BS,
        "eta_i": eta_i_Mbit
    }

if __name__ == "__main__":
    K_values = range(2000, 32000, 2000)
    aa_results = []
    acenet_results = []

    for K in K_values:
        print(f"Simulating for K = {K}")
        aa_results.append(simulate_aa(K))
        acenet_results.append(simulate_acenet(K))

    # Plotting awake BSs
    Ks = [r["K"] for r in aa_results]
    aa_awake = [r["awake_BS"] for r in aa_results]
    acenet_awake = [r["awake_BS"] for r in acenet_results]

    plt.figure()
    plt.plot(Ks, aa_awake, marker='o', label="AA (All Active)")
    plt.plot(Ks, acenet_awake, marker='s', label="ACEnet")
    plt.xlabel("# of users in high-density areas (K)")
    plt.ylabel("# of awake BS")
    plt.title("Awake BS vs User Density (AA & ACEnet)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting eta_i
    aa_etas = [r["eta_i"] for r in aa_results]
    acenet_etas = [r["eta_i"] for r in acenet_results]

    plt.figure()
    plt.plot(Ks, aa_etas, marker='o', label="AA (All Active)")
    plt.plot(Ks, acenet_etas, marker='s', label="ACEnet")
    plt.xlabel("# of users in high-density areas (K)")
    plt.ylabel("ηᵢ (J/Mbit)")
    plt.title("Energy per Information Bit vs K (AA & ACEnet)")
    plt.grid(True)
    plt.legend()
    plt.show()
