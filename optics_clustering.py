import numpy as np
from sklearn.metrics import pairwise_distances

def optics_deconet(users, min_pts=5, xi=0.1):
    """
    Implements the core steps of O-DeCoNet: OPTICS-based clustering.
    
    Args:
        users (ndarray): user coordinates, shape (N, 2)
        min_pts (int): Minimum number of points to define a core
        xi (float): steepness parameter for RD valley detection

    Returns:
        clusters: list of lists, each sublist is indices of users in one cluster
        reachability: list of reachability distances in S_ORDER
        S_ORDER: ordered list of user indices
    """
    N = users.shape[0]
    D = pairwise_distances(users)

    CD = np.full(N, np.inf)
    for i in range(N):
        sorted_dists = np.sort(D[i])
        if len(sorted_dists) > min_pts:
            CD[i] = sorted_dists[min_pts]  

    RD = np.full(N, np.inf)
    processed = np.zeros(N, dtype=bool)
    S_ORDER = []

    SSEED = set(range(N))

    while SSEED:
        # Step 3.2: pick user with smallest RD (or random if all inf)
        i = min(SSEED, key=lambda x: RD[x]) if np.isfinite(RD).any() else np.random.choice(list(SSEED))

        S_ORDER.append(i)
        SSEED.remove(i)
        processed[i] = True

        # Step 3.3–3.4: update RD for neighbors
        for k in SSEED:
            dist = D[i][k]
            TV = max(CD[i], dist)
            RD[k] = min(RD[k], TV)

    # Step 3.7: cluster detection via RD valleys
    clusters = []
    i = 0
    while i < len(S_ORDER) - 1:
        j = i
        while j < len(S_ORDER) - 1 and RD[S_ORDER[j]] * (1 - xi) >= RD[S_ORDER[j + 1]]:
            j += 1

        k = j
        while k < len(S_ORDER) - 1 and RD[S_ORDER[k]] <= RD[S_ORDER[k + 1]] * (1 - xi):
            k += 1

        if k - j >= min_pts:
            cluster = S_ORDER[j + 1:k + 1]
            clusters.append(cluster)
            i = k + 1
        else:
            i += 1

    reachability = [RD[i] for i in S_ORDER]
    return clusters, reachability, S_ORDER

