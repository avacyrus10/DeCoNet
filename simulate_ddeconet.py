import numpy as np
from generate_network import generate_network
from dbscan_clustering import dbscan_deconet
from bs_mode_control import calculate_thinning_radius, apply_mode_control, compute_power_per_area, compute_energy_per_info_bit


def main(lambda_u_high_fixed=None,
         power_params=None,
         rng_seed=None,
         visualize=False):

    """
    Simulate D-DeCoNet (DBSCAN-based) and compute η_A.

    Args:
        lambda_u_high_fixed: fixed λ_H^u for high-density areas (per km^2) or None for random.
        power_params: dict with P_tx_W, sigma, P_c_W, P_o_W (defaults if None).
        rng_seed: int for reproducibility.
        visualize: whether to plot clusters.

    Returns:
        dict: {"eta_A": float, "awake": int, "sleep": int}
    """
    if power_params is None:
        power_params = dict(P_tx_W=0.25, sigma=0.23, P_c_W=5.4, P_o_W=0.7)

    # --- Generate network ---
    BS_positions, users_all, users_low, users_high_list = generate_network(
        lambda_u_high_fixed=lambda_u_high_fixed,
        rng_seed=rng_seed
    )

    # --- Cluster users with DBSCAN ---
    merged_clusters, rc_list = dbscan_deconet(users_all)

    # --- Compute thinning radii ---
    thinning_radii = []
    for cluster_users, rc in zip(merged_clusters, rc_list):
        thinning_radii.append(
            calculate_thinning_radius(cluster_users, users_all, rc)
        )

    # --- Apply BS mode control ---
    bs_states = apply_mode_control(merged_clusters, users_all, BS_positions, thinning_radii)

    # --- Compute η_A ---
    eta_A = compute_power_per_area(bs_states, power_params, area_size_m=1000)
    
    eta_I = compute_energy_per_info_bit(bs_states, BS_positions, users_all, power_params)


    print(f"[D-DeCoNet] η_A = {eta_A:.3f} W/km^2 | η_I = {eta_I:.3e} W/bps | "
          f"AWAKE={bs_states.count('AWAKE')} SLEEP={bs_states.count('SLEEP')}")

    # --- Optional visualization ---
    if visualize == "clusters":
        from viz import plot_clusters
        plot_clusters(users_all, merged_clusters, BS_positions, thinning_radii, bs_states,
                      title="D-DeCoNet (DBSCAN) — Clusters & Thinning")


    return {
        "eta_A": eta_A,
        "eta_I": eta_I,
        "awake": bs_states.count("AWAKE"),
        "sleep": bs_states.count("SLEEP"),
    }

if __name__ == "__main__":
    main()

