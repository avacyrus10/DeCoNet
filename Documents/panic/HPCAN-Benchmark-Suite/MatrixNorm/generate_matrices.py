import numpy as np
import os
import psutil

def generate_matrices():
    """Generates dense and sparse matrices for various sizes."""
    sizes = []

    for i in range(7, 17):
        sizes.append(2 ** i)

    for i in range(8, 17):
        for j in [1, 2, 4, 6, 8]:
            sizes.append(2 ** i + j)

    sizes = sorted(set(sizes))

    os.makedirs('data', exist_ok=True)

    for size in sizes:
        available_memory = psutil.virtual_memory().available
        required_memory = size * size * 8  

        if required_memory > available_memory:
            print(f"Skipping generation of matrix {size}x{size} due to insufficient memory.")
            continue

        dense_matrix = np.random.rand(size, size).astype(np.float64)
        dense_matrix.tofile(f'data/matrix_{size}x{size}_dense.bin')
        del dense_matrix
        print(f"Generated dense matrix of size {size}x{size}")

        sparsity_levels = [0.1, 0.5, 0.9]
        for sparsity in sparsity_levels:
            sparse_matrix = np.random.rand(size, size).astype(np.float64)
            mask = np.random.rand(size, size) < sparsity
            sparse_matrix[mask] = 0
            sparse_matrix.tofile(f'data/matrix_{size}x{size}_sparse_{int(sparsity*100)}.bin')
            del sparse_matrix
            print(f"Generated sparse matrix of size {size}x{size} with sparsity {sparsity}")

if __name__ == "__main__":
    generate_matrices()

