import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from scipy.spatial import distance_matrix
from bs_mode_control import compute_energy_per_info_bit 

def simulate_aa(K):
    """
    All base stations are active.
    """
    BS_positions, users_all, _, _ = generate_network(K=K)
    awake_BS = len(BS_positions)
    eta_i = compute_energy_per_info_bit(awake_BS, len(users_all))

    return {
        "K": K,
        "awake_BS": awake_BS,
        "sleep_BS": 0,
        "total_BS": awake_BS,
        "eta_i": eta_i  
    }

def simulate_acenet(K):
    """
    Each user connects to its nearest BS. BSs with no assigned users are put to sleep.
    """
    BS_positions, users_all, _, _ = generate_network(K=K)

    # Compute distance matrix between users and BSs
    dist_matrix = distance_matrix(users_all, BS_positions)
    nearest_BS_indices = np.argmin(dist_matrix, axis=1)

    # Count users per BS
    user_count_per_BS = np.bincount(nearest_BS_indices, minlength=len(BS_positions))

    awake_BS = np.count_nonzero(user_count_per_BS)
    sleep_BS = len(BS_positions) - awake_BS
    eta_i = compute_energy_per_info_bit(awake_BS, len(users_all))

    return {
        "K": K,
        "awake_BS": awake_BS,
        "sleep_BS": sleep_BS,
        "total_BS": len(BS_positions),
        "eta_i": eta_i  
    }

if __name__ == "__main__":
    K_values = range(2000, 32000, 2000)
    aa_results = []
    acenet_results = []

    for K in K_values:
        print(f"Simulating for K = {K}")
        aa_results.append(simulate_aa(K))
        acenet_results.append(simulate_acenet(K))

    # Plotting
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

