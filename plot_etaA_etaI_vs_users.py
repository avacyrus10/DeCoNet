import numpy as np
import matplotlib.pyplot as plt
import argparse

from simulate_ddeconet import main as run_ddeconet
from simulate_odeconet import main as run_odeconet
from simulate_aa_acenet import simulate_aa, simulate_acenet

def usersK_to_lambdaHu(K_users, radius_m=100.0):
    area_km2 = (np.pi * radius_m * radius_m) / 1e6
    return K_users / area_km2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["etaA", "etaI", "both"], default="both",
                        help="which plot(s) to produce")
    args = parser.parse_args()

    K_values = [4_000, 8_000, 12_000, 16_000]
    trials = 2
    power_params = dict(P_tx_W=0.25, sigma=0.23, P_c_W=5.4, P_o_W=0.7)

    results = {
        "ddeco_etaA": [], "odeco_etaA": [], "aa_etaA": [], "acenet_etaA": [],
        "ddeco_etaI": [], "odeco_etaI": [], "aa_etaI": [], "acenet_etaI": [],
    }

    for K in K_values:
        lamHu = usersK_to_lambdaHu(K)
        etaA_d, etaI_d = [], []; etaA_o, etaI_o = [], []
        etaA_aa, etaI_aa = [], []; etaA_ac, etaI_ac = [], []

        for t in range(trials):
            res_d = run_ddeconet(lambda_u_high_fixed=lamHu, power_params=power_params,
                                 rng_seed=100 + t, visualize=False)
            res_o = run_odeconet(lambda_u_high_fixed=lamHu, power_params=power_params,
                                 rng_seed=200 + t, visualize=False)
            res_aa = simulate_aa(lambda_u_high_fixed=lamHu, power_params=power_params)
            res_ac = simulate_acenet(lambda_u_high_fixed=lamHu, power_params=power_params)

            etaA_d.append(res_d["eta_A"]); etaI_d.append(res_d["eta_I"])
            etaA_o.append(res_o["eta_A"]); etaI_o.append(res_o["eta_I"])
            etaA_aa.append(res_aa["eta_A"]); etaI_aa.append(res_aa["eta_I"])
            etaA_ac.append(res_ac["eta_A"]); etaI_ac.append(res_ac["eta_I"])

        results["ddeco_etaA"].append(np.mean(etaA_d)); results["ddeco_etaI"].append(np.mean(etaI_d))
        results["odeco_etaA"].append(np.mean(etaA_o)); results["odeco_etaI"].append(np.mean(etaI_o))
        results["aa_etaA"].append(np.mean(etaA_aa));   results["aa_etaI"].append(np.mean(etaI_aa))
        results["acenet_etaA"].append(np.mean(etaA_ac)); results["acenet_etaI"].append(np.mean(etaI_ac))

    Ks = np.array(K_values) / 1000.0

    if args.mode in ("etaA", "both"):
        plt.figure(figsize=(7, 5))
        plt.plot(Ks, results["ddeco_etaA"], 'o-', label="D-DeCoNet")
        plt.plot(Ks, results["odeco_etaA"], 's-', label="O-DeCoNet")
        plt.plot(Ks, results["aa_etaA"], '^-', label="AA")
        plt.plot(Ks, results["acenet_etaA"], 'v-', label="ACEnet")
        plt.xlabel("Users in High-Density Area (K)")
        plt.ylabel(r"$\eta_A$ (W/km$^2$)")
        plt.title(r"Power per Area Unit ($\eta_A$) vs. High-Density Users")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    if args.mode in ("etaI", "both"):
        plt.figure(figsize=(7, 5))
        plt.plot(Ks, results["ddeco_etaI"], 'o-', label="D-DeCoNet")
        plt.plot(Ks, results["odeco_etaI"], 's-', label="O-DeCoNet")
        plt.plot(Ks, results["aa_etaI"], '^-', label="AA")
        plt.plot(Ks, results["acenet_etaI"], 'v-', label="ACEnet")
        plt.xlabel("Users in High-Density Area (K)")
        plt.ylabel(r"$\eta_I$ (W/bps)")
        plt.title(r"Energy per Information Bit ($\eta_I$) vs. High-Density Users")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

