import numpy as np

def generate_network(area_size=1000, lambda_B=0.0001, lambda_u_low=0.001, lambda_u_high=0.005,
                     num_high_density_areas=3, high_density_radius=100):
    """
    Generate base stations and users using Poisson Point Process.

    Returns:
        BS_positions: ndarray of shape (num_BS, 2)
        users_all: ndarray of shape (num_users, 2)
        users_low: ndarray of low-density users
        users_high: ndarray of high-density users
    """
    # Generate BSs
    num_BS = np.random.poisson(lambda_B * area_size * area_size)
    BS_positions = np.random.uniform(0, area_size, (num_BS, 2))

    # Generate low-density users
    num_users_low = np.random.poisson(lambda_u_low * area_size * area_size)
    users_low = np.random.uniform(0, area_size, (num_users_low, 2))

    # Generate high-density clusters
    users_high = []
    for _ in range(num_high_density_areas):
        center = np.random.uniform(high_density_radius, area_size - high_density_radius, 2)
        num_users_hd = np.random.poisson(lambda_u_high * np.pi * high_density_radius ** 2)
        angles = np.random.uniform(0, 2 * np.pi, num_users_hd)
        radii = high_density_radius * np.sqrt(np.random.uniform(0, 1, num_users_hd))
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        users_high.append(np.stack((x, y), axis=1))

    users_high = np.concatenate(users_high, axis=0)
    users_all = np.concatenate([users_low, users_high], axis=0)

    return BS_positions, users_all, users_low, users_high

