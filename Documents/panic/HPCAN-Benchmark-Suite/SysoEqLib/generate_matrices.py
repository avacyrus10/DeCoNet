import numpy as np
import os

def generate_system_matrices(sizes):
    os.makedirs('data', exist_ok=True)
    for size in sizes:

        matrix = np.random.rand(size, size).astype(np.float64)

        for i in range(size):
            matrix[i, i] += size

        vector = np.random.rand(size).astype(np.float64)

        matrix.tofile(f'data/matrix_{size}x{size}.bin')
        vector.tofile(f'data/vector_{size}.bin')

if __name__ == "__main__":
    sizes = [2 ** i for i in range(7, 8)]

    for i in range(7, 8):
        for j in [1, 2, 4, 6, 8]:
            sizes.append(2 ** i + j)

    sizes.sort()

    generate_system_matrices(sizes)

