import numpy as np
import os

def generate_matrix(size):
    """Generates a random square matrix of the given size with full rank."""
    while True:
        matrix = np.random.rand(size, size).astype(np.float64)
        if np.linalg.matrix_rank(matrix) == size:
            return matrix

def store_matrices():
    """Generates and stores matrices of various sizes."""
    sizes = []

    for i in range(7, 17):
        sizes.append(2 ** i)

    for i in range(8, 17):
        for j in [1, 2, 4, 6, 8]:
            sizes.append(2 ** i + j)

    sizes = sorted(set(sizes))

    os.makedirs('data', exist_ok=True)

    for size in sizes:
        dense_matrix = generate_matrix(size)
        dense_matrix.tofile(f'data/matrix_{size}x{size}_dense.bin')
        print(f"Generated and stored matrix of size {size}x{size}")

if __name__ == "__main__":
    store_matrices()

