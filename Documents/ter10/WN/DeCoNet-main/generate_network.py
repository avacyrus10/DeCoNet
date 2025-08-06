import numpy as np

def generate_network(area_size=1000, num_high_density_areas=3,
                     lambda_B=100, lambda_u_low=1e3,
                     lambda_u_high=10e3,
                     high_density_radius=100):
    """
    Generate network based on paper simulation environment.
    
    Args:
        area_size (int): Size of the square network area in meters.
        num_high_density_areas (int): Number of clustered high-density user areas.
        lambda_B (float): BS density (per km^2).
        lambda_u_low (float): Low-density user density (per km^2).
        lambda_u_high (float): High-density user lambda value (per km^2).
        high_density_radius (int): Radius of the high-density user clusters in meters.

    Returns:
        BS_positions (ndarray): Array of BS coordinates.
        users_all (ndarray): Array of all user coordinates.
        users_low (ndarray): Array of low-density user coordinates.
        users_high (ndarray): Array of high-density user coordinates.
    """
    num_BS = int(lambda_B * (area_size ** 2) / 1e6)
    BS_positions = np.random.uniform(0, area_size, (num_BS, 2))

    num_users_low = int(lambda_u_low * (area_size ** 2) / 1e6)
    users_low = np.random.uniform(0, area_size, (num_users_low, 2))

    users_high = []
    for _ in range(num_high_density_areas):
        center = np.random.uniform(high_density_radius, area_size - high_density_radius, 2)
        num_users_hd = int(lambda_u_high * (np.pi * (high_density_radius ** 2)) / 1e6)
        
        angles = np.random.uniform(0, 2 * np.pi, num_users_hd)
        radii = high_density_radius * np.sqrt(np.random.uniform(0, 1, num_users_hd))
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        users_high.append(np.stack((x, y), axis=1))

    users_high = np.concatenate(users_high, axis=0) if users_high else np.empty((0, 2))
    users_all = np.concatenate([users_low, users_high], axis=0)

    return BS_positions, users_all, users_low, users_high
