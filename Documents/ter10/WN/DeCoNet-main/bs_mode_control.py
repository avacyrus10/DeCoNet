import numpy as np
import random

def compute_power_per_area(bs_states, BS_positions, area=1000*1000):
    """
    Calculates power per area unit (η_A) in W/km^2.
    """
    P_B_active = 33.5
    P_B_sleep = 0.7 
    
    total_power = 0
    for state in bs_states:
        if state == "AWAKE":
            total_power += P_B_active
        else:
            total_power += P_B_sleep

    return total_power / (area / 1e6)

def calculate_thinning_radius(clusters, users, BS_positions, algo_type="O-DeCoNet", reachability=None):
    """
    Calculate thinning radius (r_t) for each cluster based on algorithm type.
    """
    mu_B = 100
    
    thinning_radii = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            thinning_radii.append(0)
            continue

        if algo_type == "O-DeCoNet" and reachability is not None:
            cluster_RD = [reachability[u] for u in cluster]
            r_c = np.mean(cluster_RD)
            r_c *= 10
        else:
            cluster_coords = users[cluster]
            center = cluster_coords.mean(axis=0)
            distances = np.linalg.norm(cluster_coords - center, axis=1)
            r_c = np.percentile(distances, 90) 

        N_B_awake = len(cluster) / mu_B
        if N_B_awake <= 0:
            r_t = 0
        else:
            r_t = np.sqrt((r_c ** 2) / N_B_awake)
        thinning_radii.append(r_t)

    return thinning_radii

def apply_mode_control(clusters, users, BS_positions, thinning_radii):
    """
    Apply mode control: turn off BSs within r_t from cluster center,
    keeping one BS awake.
    """
    bs_states = ["AWAKE"] * len(BS_positions)

    for idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
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
