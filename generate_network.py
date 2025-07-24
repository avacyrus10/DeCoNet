import numpy as np

def generate_network(area_size=1000, num_high_density_areas=3,
                                lambda_B=0.1e3, lambda_u_low=1e3,
                                lambda_u_high_range=(4e3, 30e3),
                                high_density_radius=100):
    """
    Generate network based on paper simulation environment.
    Returns:
        BS_positions, users_all, users_low, users_high
    """
    # Calculate number of BSs
    num_BS = int(lambda_B * (area_size ** 2) / 1e6)  # Convert km² to m²
    BS_positions = np.random.uniform(0, area_size, (num_BS, 2))

    # Low-density users
    num_users_low = int(lambda_u_low * (area_size ** 2) / 1e6)
    users_low = np.random.uniform(0, area_size, (num_users_low, 2))

    # High-density clusters
    users_high = []
    for _ in range(num_high_density_areas):
        center = np.random.uniform(high_density_radius, area_size - high_density_radius, 2)
        # Pick random λ from the given range
        lambda_u_high = np.random.uniform(*lambda_u_high_range)
        # Expected users in high-density cluster
        num_users_hd = int(lambda_u_high * (np.pi * (high_density_radius ** 2)) / 1e6)
        angles = np.random.uniform(0, 2 * np.pi, num_users_hd)
        radii = high_density_radius * np.sqrt(np.random.uniform(0, 1, num_users_hd))
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        users_high.append(np.stack((x, y), axis=1))

    users_high = np.concatenate(users_high, axis=0)
    users_all = np.concatenate([users_low, users_high], axis=0)

    return BS_positions, users_all, users_low, users_high

