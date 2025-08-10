import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(
    users,
    clusters,
    BS_positions=None,
    thinning_radii=None,
    bs_states=None,
    title="DeCoNet Clustering with Thinning"
):
    """
    Visualize clustered users, optional thinning circles, and BS states.
    """
    plt.figure(figsize=(10, 10))

    # --- Identify unclustered users (noise) ---
    clustered_points = set()
    if clusters is not None:
        for cluster in clusters:
            for point_idx in cluster:
                clustered_points.add(point_idx)

    unclustered_indices = []
    for i in range(len(users)):
        if i not in clustered_points:
            unclustered_indices.append(i)

    if len(unclustered_indices) > 0:
        plt.scatter(
            users[unclustered_indices, 0],
            users[unclustered_indices, 1],
            c='lightgray',
            s=8,
            label='Unclustered'
        )

    # --- Plot clusters ---
    cluster_count = 0
    if clusters is not None:
        cluster_count = len(clusters)

    colormap = plt.cm.get_cmap('tab10', max(1, cluster_count))

    if clusters is not None:
        for idx, cluster in enumerate(clusters):
            pts = users[cluster]

            color = colormap(idx)
            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                s=28,
                color=color,
                label=f'Cluster {idx + 1}'
            )

            # Cluster center
            center = pts.mean(axis=0)
            plt.plot(
                center[0],
                center[1],
                marker='*',
                markersize=12,
                color='black',
                label='_nolegend_'
            )

            # Optional thinning circle
            if thinning_radii is not None:
                if idx < len(thinning_radii):
                    radius = thinning_radii[idx]
                    circle = plt.Circle(
                        center,
                        radius,
                        color=color,
                        linestyle='--',
                        alpha=0.22,
                        fill=False
                    )
                    plt.gca().add_patch(circle)

    # --- Plot base stations ---
    if BS_positions is not None:
        seen_labels = set()

        for i, bs in enumerate(BS_positions):
            # Determine state
            if isinstance(bs_states, list) and i < len(bs_states):
                state = bs_states[i]
            else:
                state = "AWAKE"

            if state == "AWAKE":
                color = 'green'
                label = 'AWAKE BS'
            else:
                color = 'red'
                label = 'SLEEP BS'

            # Avoid duplicate legend labels
            if label in seen_labels:
                label = "_nolegend_"

            plt.scatter(
                bs[0],
                bs[1],
                marker='x',
                c=color,
                s=60,
                label=label
            )

            seen_labels.add(label)

    # --- Final plot settings ---
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.35)
    plt.axis('equal')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

