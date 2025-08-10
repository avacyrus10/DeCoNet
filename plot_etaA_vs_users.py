import numpy as np
import matplotlib.pyplot as plt

from simulate_ddeconet import main as run_ddeconet
from simulate_odeconet import main as run_odeconet
from simulate_aa_acenet import simulate_aa, simulate_acenet


def usersK_to_lambdaHu(K_users, radius_m=100.0):
    """
    Convert target # of users in a high-density area (K_users)
    to λ_H^u (users per km^2).
    """
    area_km2 = (np.pi * radius_m * radius_m) / 1e6
    return K_users / area_km2


def main():

    K_values = [4_000, 8_000, 12_000, 16_000]
    trials = 2

    power_params = dict(P_tx_W=0.25, sigma=0.23, P_c_W=5.4, P_o_W=0.7)

    results_ddeco = []
    results_odeco = []
    results_aa = []
    results_acenet = []

    for K in K_values:
        lamHu = usersK_to_lambdaHu(K)  

        etaA_d = []
        etaA_o = []
        etaA_aa = []
        etaA_ac = []

        for t in range(trials):
            res_d = run_ddeconet(lambda_u_high_fixed=lamHu,
                                 power_params=power_params,
                                 rng_seed=100 + t,
                                 visualize=False)
            res_o = run_odeconet(lambda_u_high_fixed=lamHu,
                                 power_params=power_params,
                                 rng_seed=200 + t,
                                 visualize=False)
            res_aa = simulate_aa(lambda_u_high_fixed=lamHu,
                                 power_params=power_params)
            res_ac = simulate_acenet(lambda_u_high_fixed=lamHu,
                                     power_params=power_params)

            etaA_d.append(res_d["eta_A"])
            etaA_o.append(res_o["eta_A"])
            etaA_aa.append(res_aa["eta_A"])
            etaA_ac.append(res_ac["eta_A"])

        results_ddeco.append(np.mean(etaA_d))
        results_odeco.append(np.mean(etaA_o))
        results_aa.append(np.mean(etaA_aa))
        results_acenet.append(np.mean(etaA_ac))

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(np.array(K_values) / 1000.0, results_ddeco, 'o-', label="D-DeCoNet")
    plt.plot(np.array(K_values) / 1000.0, results_odeco, 's-', label="O-DeCoNet")
    plt.plot(np.array(K_values) / 1000.0, results_aa, 'd-', label="AA (All Active)")
    plt.plot(np.array(K_values) / 1000.0, results_acenet, '^-', label="ACEnet")

    plt.xlabel("Users in High-Density Area (K)")
    plt.ylabel(r"$\eta_A$ (W/km$^2$)")
    plt.title(r"Power per Area Unit ($\eta_A$) vs. High-Density Users")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

