import numpy as np
from sklearn.metrics import pairwise_distances

def odeconet_optics(users, MinPts=10, xi=0.05, min_cluster_size=50):
    from sklearn.metrics import pairwise_distances
    N = len(users)
    D = pairwise_distances(users)

    # Step 3.1: Compute Core Distances (CD)
    CD = np.full(N, np.inf)
    for i in range(N):
        sorted_distances = np.sort(D[i])
        if len(sorted_distances) > MinPts:
            CD[i] = sorted_distances[MinPts]

    # Initialize RD and order
    RD = np.full(N, np.inf)
    S_ORDER = []
    SSEED = set(range(N))

    # Steps 3.2 - 3.6: Ordering points
    while SSEED:
        if np.isfinite(RD).any():
            i = min(SSEED, key=lambda x: RD[x])
        else:
            i = np.random.choice(list(SSEED))
        S_ORDER.append(i)
        SSEED.remove(i)

        # Update RD for remaining points
        for k in SSEED:
            TVk = max(CD[i], D[i][k])
            if TVk < RD[k]:
                RD[k] = TVk

    # Step 3.7 and 3.8: Cluster extraction
    clusters = []
    i = 0
    while i < len(S_ORDER) - 1:
        j = i
        while j < len(S_ORDER) - 1 and RD[S_ORDER[j]] * (1 - xi) >= RD[S_ORDER[j + 1]]:
            j += 1

        k = j
        while k < len(S_ORDER) - 1 and RD[S_ORDER[k]] <= RD[S_ORDER[k + 1]] * (1 - xi):
            k += 1

        if (k - j) >= MinPts and (k - j) >= min_cluster_size:
            clusters.append(S_ORDER[j + 1:k + 1])

        i = k + 1

    return clusters, RD, S_ORDER


