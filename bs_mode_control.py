from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial import cKDTree
import math

AWAKE = "AWAKE"
SLEEP = "SLEEP"


def calculate_thinning_radius(
    cluster_users: np.ndarray,
    users_all: np.ndarray,
    rc: float,
    mu_B: int = 100
) -> float:
    """
    Compute thinning radius r_t for a cluster using:
        r_t = sqrt( (r_c^2) / N_B_awake ),
    where N_B_awake = (#users in cluster) / mu_B.
    """
    # Number of users in the cluster 
    n_users = int(len(cluster_users))
    if n_users < 1:
        n_users = 1

    # Number of awake BSs for this cluster 
    n_bs_awake = n_users / float(mu_B)
    if n_bs_awake < 1.0:
        n_bs_awake = 1.0

    # r_t formula
    numerator = rc ** 2
    rt = np.sqrt(numerator / n_bs_awake)

    return float(rt)


def apply_mode_control(
    clusters: List[np.ndarray],
    users_all: np.ndarray,
    bs_positions: np.ndarray,
    thinning_radii: List[float],
    initial_state: str = AWAKE
) -> List[str]:
    """
    - Start with all BSs AWAKE.
    - For each cluster i, randomly choose an AWAKE BS as a 'reference';
      set any other AWAKE BSs within r_t(i) of that BS to SLEEP.
    """
    N = len(bs_positions)
    states = [initial_state] * N
    rng = np.random.default_rng()

    if len(clusters) == 0:
        return states

    for i, rt in enumerate(thinning_radii):
        # Collect indices of currently AWAKE BSs
        awake_idxs = []
        for idx, s in enumerate(states):
            if s == AWAKE:
                awake_idxs.append(idx)

        if len(awake_idxs) == 0:
            break

        # Pick a random reference among AWAKE BSs
        ref = int(rng.choice(awake_idxs))
        ref_pos = bs_positions[ref]

        # Distances from reference BS to all BSs
        diffs = bs_positions - ref_pos
        dists = np.linalg.norm(diffs, axis=1)

        # Put nearby AWAKE BSs to sleep (except the reference itself)
        for j in awake_idxs:
            if j == ref:
                continue

            dist_to_ref = dists[j]
            if dist_to_ref < rt:
                states[j] = SLEEP

    return states


def compute_power_per_area(
    bs_states: List[str],
    power_params: Dict[str, float],
    area_size_m: float = 1000.0
) -> float:
    """
    Compute η_A (power per area unit) in W/km^2.

    η_A = ( #AWAKE * P_B ) / Area_km2
    P_B = (1/σ) * P_tx + P_c + P_o
    """
    # Count AWAKE BSs
    awake = 0
    for s in bs_states:
        if s == AWAKE:
            awake += 1

    # Power params (with defaults)
    if "P_tx_W" in power_params:
        P_tx = power_params["P_tx_W"]
    else:
        P_tx = 0.25

    if "sigma" in power_params:
        sigma = power_params["sigma"]
    else:
        sigma = 0.23

    if "P_c_W" in power_params:
        P_c = power_params["P_c_W"]
    else:
        P_c = 5.4

    if "P_o_W" in power_params:
        P_o = power_params["P_o_W"]
    else:
        P_o = 0.7

    # Per-BS power and total power
    P_B = (1.0 / sigma) * P_tx + P_c + P_o
    total_power_W = float(awake) * P_B

    # Convert square meters to square kilometers
    area_km2 = (area_size_m / 1000.0) ** 2
    eta_A = total_power_W / area_km2

    return float(eta_A)


def compute_energy_per_info_bit(
    bs_states: List[str],
    bs_positions: np.ndarray,
    users_all: np.ndarray,
    power_params: Dict[str, float],
    bandwidth_hz: float = 10e6,
    pathloss_alpha: float = 3.5,
    noise_figure_dB: float = 0.0
) -> float:
    """
    Compute η_I (energy per information bit) in W/bps:
        η_I = (sum P_B over AWAKE BSs) / (sum_user B * log2(1 + SINR))
    """
    # Power params (with defaults)
    if "P_tx_W" in power_params:
        P_tx = power_params["P_tx_W"]
    else:
        P_tx = 0.25

    if "sigma" in power_params:
        sigma = power_params["sigma"]
    else:
        sigma = 0.23

    if "P_c_W" in power_params:
        P_c = power_params["P_c_W"]
    else:
        P_c = 5.4

    if "P_o_W" in power_params:
        P_o = power_params["P_o_W"]
    else:
        P_o = 0.7

    P_B = (1.0 / sigma) * P_tx + P_c + P_o

    # Indices of AWAKE BSs
    awake_idx_list = []
    for i, s in enumerate(bs_states):
        if s == AWAKE:
            awake_idx_list.append(i)
    awake_idx = np.array(awake_idx_list, dtype=int)

    if awake_idx.size == 0 or users_all.size == 0:
        return float("inf")

    # Positions of AWAKE BSs
    awake_pos = bs_positions[awake_idx]

    tree = cKDTree(awake_pos)
    d_serv, nn_idx = tree.query(users_all, k=1)

    # Avoid zero distance
    d_serv = np.clip(d_serv, 1.0, None)

    # Map to global BS indices
    serving_global = awake_idx[nn_idx]

    # Received signal from serving BS
    S = P_tx * (d_serv ** (-pathloss_alpha))


    I = np.zeros_like(S)
    for j in range(len(awake_pos)):
        pos = awake_pos[j]

        # Distances from users to this awake BS 
        diffs = users_all - pos
        d = np.linalg.norm(diffs, axis=1)
        d = np.clip(d, 1.0, None)

        # Power received from this BS
        p = P_tx * (d ** (-pathloss_alpha))

        # Exclude this BS's contribution for users it serves
        this_global_idx = awake_idx[j]
        mask = serving_global != this_global_idx
        I[mask] += p[mask]

    # Thermal noise over bandwidth
    N0_dBm_per_Hz = -174.0 + noise_figure_dB
    ten_log_B = 10.0 * math.log10(bandwidth_hz)
    N_dBm = N0_dBm_per_Hz + ten_log_B
    N_W = 10 ** ((N_dBm - 30.0) / 10.0)

    # SINR per user
    denom = I + N_W
    SINR = S / denom

    # Sum throughput over all users
    one_plus_sinr = 1.0 + SINR
    spectral_eff = np.log2(one_plus_sinr)
    rate_per_user_bps = bandwidth_hz * spectral_eff
    rate_sum_bps = float(np.sum(rate_per_user_bps))

    if rate_sum_bps <= 0.0:
        return float("inf")

    # Total consumed power by AWAKE BSs
    total_power_W = float(len(awake_idx)) * P_B

    eta_I = total_power_W / rate_sum_bps
    return float(eta_I)

