import numpy as np
import random

def calculate_thinning_radius(clusters, users, BS_positions, algo_type="O-DeCoNet", reachability=None):
    """
    Calculate thinning radius (r_t) for each cluster based on algorithm type.

    Args:
        clusters (list of lists): User indices for each cluster
        users (ndarray): All user coordinates
        BS_positions (ndarray): Base station coordinates
        algo_type (str): "O-DeCoNet" or "D-DeCoNet"
        reachability (list): Reachability distances (only for O-DeCoNet)

    Returns:
        thinning_radii (list): r_t for each cluster
    """
    total_users = len(users)
    total_BS = len(BS_positions)
    mu_B = total_users / total_BS  # Average users per BS

    thinning_radii = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            thinning_radii.append(0)
            continue

        # Compute r_c^i
        if algo_type == "O-DeCoNet" and reachability is not None:
            cluster_RD = [reachability[u] for u in cluster]
            r_c = np.mean(cluster_RD)
        else:  # D-DeCoNet
            cluster_coords = users[cluster]
            center = cluster_coords.mean(axis=0)
            distances = np.linalg.norm(cluster_coords - center, axis=1)
            r_c = np.max(distances)  # furthest point from center

        # Compute N_B^i,awake
        N_B_awake = len(cluster) / mu_B

        # Compute thinning radius r_t^i
        r_t = np.sqrt((r_c ** 2) / N_B_awake)
        thinning_radii.append(r_t)

    return thinning_radii


def apply_mode_control(clusters, users, BS_positions, thinning_radii):
    """
    Apply mode control: turn off BSs within r_t from cluster center,
    keeping one BS awake.

    Returns:
        bs_states: list of "AWAKE" or "SLEEP"
    """
    bs_states = ["AWAKE"] * len(BS_positions)

    for idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue

        cluster_coords = users[cluster]
        cluster_center = cluster_coords.mean(axis=0)
        r_t = thinning_radii[idx]

        # Find BSs within r_t
        distances = np.linalg.norm(BS_positions - cluster_center, axis=1)
        candidate_BSs = [i for i, d in enumerate(distances) if d <= r_t]

        if len(candidate_BSs) > 1:
            # Randomly keep one awake, others sleep
            keep_awake = random.choice(candidate_BSs)
            for i in candidate_BSs:
                if i != keep_awake:
                    bs_states[i] = "SLEEP"

    return bs_states

