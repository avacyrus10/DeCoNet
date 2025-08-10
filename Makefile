PY ?= python3
K  ?= 6000
RNG ?= 123

.PHONY: help etaA etaI both viz-d viz-o clean

help:
	@echo "Targets:"
	@echo "  make etaA     -> plot η_A (power per area)"
	@echo "  make etaI     -> plot η_I only"
	@echo "  make both     -> plot both η_A and η_I"
	@echo "  make viz-d    -> visualize D-DeCoNet clusters at K=$(K), RNG=$(RNG)"
	@echo "  make viz-o    -> visualize O-DeCoNet clusters at K=$(K), RNG=$(RNG)"
	@echo "  make clean    -> remove caches"

etaA:
	$(PY) plot_etaA_vs_users.py

etaI:
	$(PY) plot_etaA_etaI_vs_users.py --mode etaI

both:
	$(PY) plot_etaA_etaI_vs_users.py --mode both

viz-d:
	$(PY) -c "from simulate_ddeconet import main as run_d; from plot_etaA_etaI_vs_users import usersK_to_lambdaHu as f; lam=f(int($(K))); run_d(lambda_u_high_fixed=lam, rng_seed=$(RNG), visualize='clusters')"

viz-o:
	$(PY) -c "from simulate_odeconet import main as run_o; from plot_etaA_etaI_vs_users import usersK_to_lambdaHu as f; lam=f(int($(K))); run_o(lambda_u_high_fixed=lam, rng_seed=$(RNG), visualize='clusters')"

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

