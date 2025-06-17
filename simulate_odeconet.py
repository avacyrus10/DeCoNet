import matplotlib.pyplot as plt
from generate_network import generate_network
from optics_clustering import optics_deconet


def plot_clusters(users, clusters, BS_positions=None, title="O-DeCoNet Clustering"):
    plt.figure(figsize=(10, 10))


    plt.scatter(users[:, 0], users[:, 1], c='lightgray', s=10, label='Unclustered Users')

    colors = plt.cm.get_cmap('tab10', len(clusters))
    for idx, cluster in enumerate(clusters):
        cluster_coords = users[cluster]
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=20, label=f'Cluster {idx+1}', color=colors(idx))

    if BS_positions is not None:
        plt.scatter(BS_positions[:, 0], BS_positions[:, 1], marker='x', c='black', label='Base Stations')

    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_reachability(reachability):
    plt.figure(figsize=(10, 4))
    plt.plot(reachability, color='darkblue')
    plt.title("Reachability Distance (RD) Plot - O-DeCoNet")
    plt.xlabel("Order of users (S_ORDER)")
    plt.ylabel("Reachability Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():

    BS_positions, users_all, _, _ = generate_network()

    min_pts = 3
    xi = 0.2
    clusters, reachability, order = optics_deconet(users_all, min_pts=min_pts, xi=xi)

    print(f"Total clusters found: {len(clusters)}")
    print("Sample reachability distances:", reachability[:10])

    plot_clusters(users_all, clusters, BS_positions, title="O-DeCoNet (OPTICS-Based) Clustering of Users")
    plot_reachability(reachability)


if __name__ == "__main__":
    main()

