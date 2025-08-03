import matplotlib.pyplot as plt

# Import simulation methods
from simulate_odeconet import main as run_odeconet
from simulate_ddeconet import main as run_ddeconet
from simulate_aa_acenet import simulate_aa, simulate_acenet

# Define K range
K_values = [2000, 4000, 6000]


# Run simulations
results_odeco = [run_odeconet(K=K, visualize=False) for K in K_values]
results_ddeco = [run_ddeconet(K=K, visualize=False) for K in K_values]
results_aa = [simulate_aa(K) for K in K_values]
results_acenet = [simulate_acenet(K) for K in K_values]

# Extract ηᵢ values
eta_odeco = [r["eta_i"] for r in results_odeco]
eta_ddeco = [r["eta_i"] for r in results_ddeco]
eta_aa = [r["eta_i"] for r in results_aa]
eta_acenet = [r["eta_i"] for r in results_acenet]

# Plot ηᵢ vs K
plt.figure(figsize=(10, 6))
plt.plot(K_values, eta_odeco, marker='o', label="O-DeCoNet")
plt.plot(K_values, eta_ddeco, marker='^', label="D-DeCoNet")
plt.plot(K_values, eta_acenet, marker='s', label="ACEnet")
plt.plot(K_values, eta_aa, marker='x', label="AA")

plt.xlabel("Number of users in high-density areas (K)")
plt.ylabel("Energy per Information Bit ηᵢ (J/Mbit)")
plt.title("Comparison of ηᵢ vs K for All Algorithms")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

