import numpy as np
import random
from collections import defaultdict
from scipy.spatial import distance_matrix

def calculate_thinning_radius(clusters, users, BS_positions, algo_type="O-DeCoNet", reachability=None):
    """
    Calculates the thinning radius for each cluster based on the algorithm type.
    """
    mu_B = 100  

    thinning_radii = []
    for idx, cluster in enumerate(clusters):
        if not cluster:
            thinning_radii.append(0)
            continue

        if algo_type == "O-DeCoNet" and reachability is not None:
            cluster_RD = [reachability[u] for u in cluster]
            r_c = np.mean(cluster_RD)
        else:
            cluster_coords = users[cluster]
            center = cluster_coords.mean(axis=0)
            distances = np.linalg.norm(cluster_coords - center, axis=1)
            r_c = np.max(distances)

        N_B_awake = max(1.0, len(cluster) / mu_B)
        r_t = np.sqrt((r_c ** 2) / N_B_awake)
        thinning_radii.append(r_t)

    return thinning_radii

def apply_mode_control_multi(clusters, users, BS_positions, thinning_radii):
    """
    Implements the BS mode control with a fallback to ensure connectivity.
    For each cluster, a single BS is chosen from candidates within the thinning radius.
    If no candidates are found, the closest BS is selected as a fallback.
    """
    bs_states = ["SLEEP"] * len(BS_positions)
    assigned_bs = set()
    
    for idx, cluster in enumerate(clusters):
        if not cluster:
            continue
            
        cluster_coords = users[cluster]
        cluster_center = cluster_coords.mean(axis=0)
        r_t = thinning_radii[idx]

        distances = np.linalg.norm(BS_positions - cluster_center, axis=1)
        candidate_bs_indices = [i for i, d in enumerate(distances) if d <= r_t]

        unassigned_candidates = list(set(candidate_bs_indices) - assigned_bs)

        if len(unassigned_candidates) > 0:
            keep_awake_idx = random.choice(unassigned_candidates)
        else:
            sorted_bs_indices = np.argsort(distances)
            closest_unassigned_bs = next((i for i in sorted_bs_indices if i not in assigned_bs), None)
            
            if closest_unassigned_bs is not None:
                keep_awake_idx = closest_unassigned_bs
            else:
                continue

        bs_states[keep_awake_idx] = "AWAKE"
        assigned_bs.add(keep_awake_idx)
            
    return bs_states


def compute_sinr(users, bs_states, BS_positions, P_tx=0.25, pathloss_exp=3.5, bandwidth=10e6):
    """
    Computes the SINR for each user.
    """
    users = np.array(users)
    BS_positions = np.array(BS_positions)
    sinr_dict = defaultdict(list)

    awake_indices = [i for i, state in enumerate(bs_states) if state == "AWAKE"]
    
    if not awake_indices:
        return {}
    
    awake_positions = BS_positions[awake_indices]
    
    # === ADJUSTED NOISE POWER CALCULATION ===
    # This value is adjusted to match the scale of the paper's results more closely.
    # A higher noise power will reduce the SINR and increase ηᵢ.
    noise_power = 1e-10  
    
    for user in users:
        distances = np.linalg.norm(awake_positions - user, axis=1)
        min_idx = np.argmin(distances)
        serving_bs_global_idx = awake_indices[min_idx]
        d_serving = distances[min_idx]

        h_serving = 1 / (d_serving ** pathloss_exp + 1e-12)
        signal_power = P_tx * h_serving

        interference = 0.0
        for j, other_bs_idx in enumerate(awake_indices):
            if other_bs_idx == serving_bs_global_idx:
                continue
            d_interfere = np.linalg.norm(user - BS_positions[other_bs_idx])
            h_interfere = 1 / (d_interfere ** pathloss_exp + 1e-12)
            interference += P_tx * h_interfere

        sinr = signal_power / (interference + noise_power)
        sinr = min(sinr, 1000)
        sinr_dict[serving_bs_global_idx].append(sinr)

    return dict(sinr_dict)


def compute_energy_per_info_bit(
    sinr_dict,
    P_tx=0.25,       # W
    sigma=0.23,      # amplifier efficiency
    P_c=5.4,         # W, circuit
    P_o=0.7,         # W, standby/overhead
    bandwidth=10e6,  # Hz per BS (total, shared among users)
    cap_sinr_db=15,  # cap SINR to avoid extreme rates
    verbose=True
):
    """
    Implements Eq. (11) from the paper: ηᵢ = Total Power / Total Rate.
    Returns ηᵢ in Joules/bit.
    """
    P_B = (P_tx / sigma) + P_c + P_o
    num_awake_BS = len(sinr_dict)
    total_power = num_awake_BS * P_B

    if verbose:
        print(f"Total AWAKE BSs: {num_awake_BS}")
        print(f"Power per BS (W): {P_tx/sigma:.4f} + {P_c:.4f} + {P_o:.4f} = {P_B:.4f} W")

    total_rate = 0.0
    cap_lin = 10 ** (cap_sinr_db / 10.0) if cap_sinr_db is not None else None

    for bs_idx, sinrs in sinr_dict.items():
        N_i = len(sinrs)
        if N_i == 0:
            continue

        sinrs = np.asarray(sinrs, dtype=float)
        if cap_lin is not None:
            sinrs = np.minimum(sinrs, cap_lin)

        user_bw = bandwidth / N_i
        bs_rate = np.sum(user_bw * np.log2(1.0 + sinrs))

        total_rate += bs_rate

        if verbose:
            mean_se = np.mean(np.log2(1.0 + sinrs))
            print(f"BS {bs_idx:3d}: users={N_i:4d}, mean log2(1+SINR)={mean_se:5.3f} b/s/Hz, "
                  f"rate={bs_rate/1e6:8.2f} Mbps")

    if total_rate <= 0:
        if verbose:
            print("Total rate is zero => eta_I = infinity")
        return float('inf')

    if verbose:
        print(f"Total Energy Consumption: {total_power:.4f} J/s")
        print(f"Total Information Rate: {total_rate:.4f} bps")

    eta_I = total_power / total_rate
    if verbose:
        print(f"eta_I (Joules/bit): {eta_I:.6f}")

    return eta_I
