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
    sizes = [128, 256, 512]
    generate_system_matrices(sizes)

