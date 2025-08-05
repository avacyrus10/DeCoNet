import numpy as np

def generate_network(area_size=1000, num_high_density_areas=3,
                     lambda_B=0.1e3, lambda_u_low=1e3,
                     K=None, lambda_u_high_range=(4e3, 30e3),
                     high_density_radius=100, fixed_bs=False, seed=None):
    """
    Generate network based on paper's Table 1.

    Args:
        area_size: length of network square (m)
        num_high_density_areas: number of high-density user areas
        lambda_B: BS density (per km²)
        lambda_u_low: low-density user density (per km²)
        K: fixed number of users per high-density area (if None, random density is used)
        lambda_u_high_range: range of λ for high-density users (per km²), used if K is None
        high_density_radius: radius of each high-density area (m)
        fixed_bs: if True, deploy BSs in fixed hexagonal grid; else random uniform
        seed: optional RNG seed for reproducibility

    Returns:
        BS_positions, users_all, users_low, users_high
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Base Stations ---
    num_BS = int(lambda_B * (area_size ** 2) / 1e6)  # per km² 
    if fixed_bs:
        # Simple square grid 
        grid_size = int(np.sqrt(num_BS))
        xs = np.linspace(0, area_size, grid_size)
        ys = np.linspace(0, area_size, grid_size)
        BS_positions = np.array([(x, y) for x in xs for y in ys])
        BS_positions = BS_positions[:num_BS]
    else:
        BS_positions = np.random.uniform(0, area_size, (num_BS, 2))

    # --- Low-density users ---
    num_users_low = int(lambda_u_low * (area_size ** 2) / 1e6)
    users_low = np.random.uniform(0, area_size, (num_users_low, 2))

    # --- High-density users ---
    users_high = []
    for _ in range(num_high_density_areas):
        center = np.random.uniform(high_density_radius, area_size - high_density_radius, 2)

        if K is not None:
            num_users_hd = K
        else:
            lambda_u_high = np.random.uniform(*lambda_u_high_range)
            area_hd = np.pi * (high_density_radius ** 2)
            num_users_hd = int(lambda_u_high * area_hd / 1e6)

        angles = np.random.uniform(0, 2 * np.pi, num_users_hd)
        radii = high_density_radius * np.sqrt(np.random.uniform(0, 1, num_users_hd))
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        users_hd = np.stack((x, y), axis=1)
        users_high.append(users_hd)

    users_high = np.concatenate(users_high, axis=0)
    users_all = np.concatenate([users_low, users_high], axis=0)

    return BS_positions, users_all, users_low, users_high

