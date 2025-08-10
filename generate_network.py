import numpy as np

def generate_network(
    area_size_m: float = 1000.0,
    lambda_B: float = 0.1e3,  # BS density per km^2
    num_high_density_areas: int = 3,
    lambda_u_low: float = 1e3,     # low-density user intensity per km^2
    lambda_u_high_range=(4e3, 30e3),  # high-density user intensity range per km^2
    high_density_radius: float = 100.0, 
    rng_seed: int | None = None,
    lambda_u_high_fixed: float | None = None,  
):
    """
    Generates BS and user positions.

    BSs: PPP with λ_B (per km^2) over area_size_m^2.
    Users:
      - Low-density users: PPP with λ_L^u (per km^2) over the entire area.
      - High-density users: For each of num_high_density_areas disks of radius R,
        PPP with λ_H^u inside the disk (either fixed or sampled from range).

    Returns:
        BS_positions: (N_B, 2)
        users_all: (N_U, 2)
        users_low: (N_L, 2)
        users_high: list of arrays, one per high-density area
    """
    rng = np.random.default_rng(rng_seed)

    # ---- Helpers ----
    def ppp_points(lambda_per_km2, size_m):
        area_km2 = (size_m / 1000.0) ** 2
        mean_n = lambda_per_km2 * area_km2
        n = rng.poisson(mean_n)
        pts = rng.uniform(0.0, size_m, size=(n, 2))
        return pts

    def ppp_points_in_disk(center, radius_m, lambda_per_km2, size_m):
        area_km2 = (np.pi * radius_m * radius_m) / 1e6
        n = rng.poisson(lambda_per_km2 * area_km2)
        # Sample radius with sqrt for uniform
        r = radius_m * np.sqrt(rng.uniform(size=n))
        theta = rng.uniform(0, 2 * np.pi, size=n)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        # clip to bounds
        x = np.clip(x, 0.0, size_m)
        y = np.clip(y, 0.0, size_m)
        return np.stack([x, y], axis=1)

    # ---- BSs ----
    BS_positions = ppp_points(lambda_B, area_size_m)

    # ---- Low-density users over whole area ----
    users_low = ppp_points(lambda_u_low, area_size_m)

    # ---- High-density areas ----
    users_high_list = []
    for _ in range(num_high_density_areas):
        margin = high_density_radius
        cx = rng.uniform(margin, area_size_m - margin)
        cy = rng.uniform(margin, area_size_m - margin)
        center = np.array([cx, cy])

        if lambda_u_high_fixed is not None:
            lamH = float(lambda_u_high_fixed)
        else:
            lamH = rng.uniform(*lambda_u_high_range)

        users_h = ppp_points_in_disk(center, high_density_radius, lamH, area_size_m)
        users_high_list.append(users_h)

    # ---- Aggregate ----
    users_all = users_low.copy()
    if users_high_list:
        users_all = np.vstack([users_all] + users_high_list)

    return BS_positions, users_all, users_low, users_high_list

