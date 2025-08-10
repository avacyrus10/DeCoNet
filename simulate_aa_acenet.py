import numpy as np
from scipy.spatial import distance_matrix
from generate_network import generate_network
from bs_mode_control import compute_power_per_area, compute_energy_per_info_bit


DEFAULT_POWER = dict(P_tx_W=0.25, sigma=0.23, P_c_W=5.4, P_o_W=0.7)

def simulate_aa(lambda_u_high_fixed,
                power_params=None,
                rng_seed=None,
                visualize=False):
    """
    AA baseline: all BSs are awake.
    lambda_u_high_fixed: λ_H^u (users/km^2) for the high-density area (single area).
    """
    if power_params is None:
        power_params = DEFAULT_POWER

    BS_positions, users_all, _, _ = generate_network(
        lambda_u_high_fixed=lambda_u_high_fixed,
        lambda_u_low=0,
        num_high_density_areas=1,
        rng_seed=rng_seed
    )

    bs_states = ["AWAKE"] * len(BS_positions)
    eta_A = compute_power_per_area(bs_states, power_params, area_size_m=1000.0)
    eta_I = compute_energy_per_info_bit(bs_states, BS_positions, users_all, power_params)

    return {
        "eta_A": eta_A,
        "eta_I": eta_I,
        "awake_BS": len(BS_positions),
        "sleep_BS": 0,
    }

def simulate_acenet(lambda_u_high_fixed,
                    power_params=None,
                    rng_seed=None,
                    visualize=False):
    """
    ACEnet baseline: users attach to nearest BS; BSs with no users sleep.
    """
    if power_params is None:
        power_params = DEFAULT_POWER

    BS_positions, users_all, _, _ = generate_network(
        lambda_u_high_fixed=lambda_u_high_fixed,
        lambda_u_low=0,
        num_high_density_areas=1,
        rng_seed=rng_seed
    )

    num_BS = len(BS_positions)
    if users_all.shape[0] == 0 or num_BS == 0:
        bs_states = ["SLEEP"] * num_BS
        eta_A = compute_power_per_area(bs_states, power_params, area_size_m=1000.0)
        return {"eta_A": eta_A, "awake_BS": 0, "sleep_BS": num_BS}

    # Nearest-BS assignment
    dmat = distance_matrix(users_all, BS_positions)
    nearest_idx = np.argmin(dmat, axis=1)
    awake_set = np.unique(nearest_idx)

    bs_states = ["SLEEP"] * num_BS
    for idx in awake_set:
        bs_states[idx] = "AWAKE"

    eta_A = compute_power_per_area(bs_states, power_params, area_size_m=1000.0)
    eta_I = compute_energy_per_info_bit(bs_states, BS_positions, users_all, power_params)

    return {
        "eta_A": eta_A,
        "eta_I": eta_I,
        "awake_BS": int(len(awake_set)),
        "sleep_BS": int(num_BS - len(awake_set)),
    }

