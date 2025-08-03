import numpy as np
import random

def calculate_thinning_radius(clusters, users, BS_positions, algo_type="O-DeCoNet", reachability=None):
    """
    Calculate thinning radius (r_t) for each cluster based on algorithm type.

    Args:
        clusters (list of list): Each sublist contains indices of users in one cluster
        users (ndarray): Coordinates of all users
        BS_positions (ndarray): Coordinates of all base stations
        algo_type (str): "O-DeCoNet" or "D-DeCoNet"
        reachability (list): Reachability distances from OPTICS (for O-DeCoNet only)

    Returns:
        thinning_radii (list): r_t for each cluster
    """
    total_users = len(users)
    total_BS = len(BS_positions)
    mu_B = total_users / total_BS  # Average users per BS

    thinning_radii = []

    for idx, cluster in enumerate(clusters):
        if not cluster:
            thinning_radii.append(0)
            continue

        if algo_type == "O-DeCoNet" and reachability is not None:
            cluster_RD = [reachability[u] for u in cluster]
            r_c = np.mean(cluster_RD)
            r_c *= 3  # Empirical multiplier to scale radius
        else:
            cluster_coords = users[cluster]
            center = cluster_coords.mean(axis=0)
            distances = np.linalg.norm(cluster_coords - center, axis=1)
            r_c = np.max(distances)

        N_B_awake = len(cluster) / mu_B
        r_t = np.sqrt((r_c ** 2) / N_B_awake)
        thinning_radii.append(r_t)

    return thinning_radii


def apply_mode_control(clusters, users, BS_positions, thinning_radii):
    """
    Apply mode control logic: put BSs within r_t of a cluster center to SLEEP, except one.

    Args:
        clusters (list of list): User indices per cluster
        users (ndarray): Coordinates of users
        BS_positions (ndarray): Coordinates of BSs
        thinning_radii (list): Thinning radius per cluster

    Returns:
        bs_states (list of str): One of "AWAKE" or "SLEEP" for each BS
    """
    bs_states = ["AWAKE"] * len(BS_positions)

    for idx, cluster in enumerate(clusters):
        if not cluster:
            continue

        cluster_coords = users[cluster]
        cluster_center = cluster_coords.mean(axis=0)
        r_t = thinning_radii[idx]

        distances = np.linalg.norm(BS_positions - cluster_center, axis=1)
        candidate_BSs = [i for i, d in enumerate(distances) if d <= r_t]

        if len(candidate_BSs) > 1:
            keep_awake = random.choice(candidate_BSs)
            for i in candidate_BSs:
                if i != keep_awake:
                    bs_states[i] = "SLEEP"

    return bs_states


def compute_energy_per_info_bit(num_awake_BS, total_users, power_per_BS=500, avg_throughput_per_user=1):
    """
    Compute ηᵢ = (Total Energy Consumption) / (Total Information Transmitted)

    Args:
        num_awake_BS (int): Number of BSs in AWAKE state
        total_users (int): Total number of users in the network
        power_per_BS (float): Power consumption per BS (Watts)
        avg_throughput_per_user (float): Average throughput per user (Mbps)

    Returns:
        float: ηᵢ in Joules/Mbit
    """
    energy_total = num_awake_BS * power_per_BS  # Joules/sec
    info_total = total_users * avg_throughput_per_user  # Mbit/sec

    if info_total == 0:
        return float('inf')

    return energy_total / info_total  # J/Mbit

