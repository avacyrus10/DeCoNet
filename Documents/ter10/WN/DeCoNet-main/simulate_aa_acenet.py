import numpy as np
import matplotlib.pyplot as plt
from generate_network import generate_network
from scipy.spatial import distance_matrix
from bs_mode_control import compute_power_per_area

def simulate_aa(K_lambda):
    """
    All base stations are active.
    
    Args:
        K_lambda (float): The lambda value for high-density users.
    """
    BS_positions, users_all, _, _ = generate_network(lambda_u_high=K,
                                                 lambda_u_low=0,
                                                 num_high_density_areas=1)

    
    # All BSs are awake
    bs_states = ["AWAKE"] * len(BS_positions)
    
    eta_a = compute_power_per_area(bs_states, BS_positions)
    return {
        "K_lambda": K_lambda,
        "awake_BS": len(BS_positions),
        "sleep_BS": 0,
        "total_BS": len(BS_positions),
        "eta_a": eta_a
    }

def simulate_acenet(K_lambda):
    """
    Each user connects to its nearest BS. BSs with no assigned users are put to sleep.
    
    Args:
        K_lambda (float): The lambda value for high-density users.
    """
    BS_positions, users_all, _, _ = generate_network(lambda_u_high=K,
                                                 lambda_u_low=0,
                                                 num_high_density_areas=1)

    num_BS = len(BS_positions)
    
    # Compute distance matrix between users and BSs
    dist_matrix = distance_matrix(users_all, BS_positions)
    
    # Find the nearest BS for each user
    nearest_BS_indices = np.argmin(dist_matrix, axis=1)

    # Determine which BSs are awake
    awake_bs_indices = np.unique(nearest_BS_indices)
    bs_states = ["SLEEP"] * num_BS
    for idx in awake_bs_indices:
        bs_states[idx] = "AWAKE"
    
    eta_a = compute_power_per_area(bs_states, BS_positions)
    awake_BS = len(awake_bs_indices)
    sleep_BS = num_BS - awake_BS
    
    return {
        "K_lambda": K_lambda,
        "awake_BS": awake_BS,
        "sleep_BS": sleep_BS,
        "total_BS": num_BS,
        "eta_a": eta_a
    }

if __name__ == "__main__":
    # K_values_plot represents the x-axis values (K) in thousands
    K_values_plot = range(2, 32, 2)
    # lambda_u_high_values is the actual lambda value for the simulation
    lambda_u_high_values = [k * 1000 for k in K_values_plot]
    
    aa_results = []
    acenet_results = []

    for K_lambda in lambda_u_high_values:
        print(f"Simulating for K = {K_lambda/1000}")
        aa_results.append(simulate_aa(K_lambda))
        acenet_results.append(simulate_acenet(K_lambda))

    # Plotting eta_A
    Ks = [r["K_lambda"]/1000 for r in aa_results]
    aa_etas = [r["eta_a"] for r in aa_results]
    acenet_etas = [r["eta_a"] for r in acenet_results]

    plt.figure()
    plt.plot(Ks, aa_etas, marker='o', label="AA (All Active)")
    plt.plot(Ks, acenet_etas, marker='s', label="ACEnet")
    plt.xlabel("# of users in high-density areas (K)")
    plt.ylabel("Power per area unit ($W/km^2$)")
    plt.title("Energy Efficiency ($\eta_A$) vs. High User-Density (AA & ACEnet)")
    plt.grid(True)
    plt.legend()
    plt.show()
