import numpy as np
import subprocess
import os

def load_3d_array(filename, shape):
    """Loads a 3D matrix from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    return np.loadtxt(filename).reshape(shape)

def matrix_sum_3d(A, B):
    """Computes the sum of two 3D matrices."""
    return A + B

def validate(sizes, sparsities, data_dir, use_cublas=False):
    """Validates the CUDA implementation by comparing outputs."""
    for width, height, depth in sizes:
        for sparsity in sparsities:
            matrixA_file = os.path.join(data_dir, f"matrixA_{width}x{height}x{depth}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{width}x{height}x{depth}_{sparsity}.txt")
            output_file = os.path.join(data_dir, f"output_{width}x{height}x{depth}_{sparsity}.txt")

            exec_bin = "./bin/matrixSum3D_cublas" if use_cublas else "./bin/matrixSum3D"

            result = subprocess.run([exec_bin, matrixA_file, matrixB_file, output_file, str(width), str(height), str(depth)])

            if result.returncode != 0:
                print(f"Error running {exec_bin}. Return code: {result.returncode}")
                continue

            try:
                output_data = load_3d_array(output_file, (depth, height, width))
            except FileNotFoundError as e:
                print(f"Error: {e}")
                continue

            matrixA = load_3d_array(matrixA_file, (depth, height, width))
            matrixB = load_3d_array(matrixB_file, (depth, height, width))

            expected_output = matrix_sum_3d(matrixA, matrixB)
            tolerance = 1e-3
            if np.allclose(output_data, expected_output, atol=tolerance):
                print(f"Test passed for size {width}x{height}x{depth}, sparsity={sparsity}")
            else:
                print(f"Test failed for size {width}x{height}x{depth}, sparsity={sparsity}")

if __name__ == "__main__":
    sizes = []
    for i in range(7, 17):  
        size = 2 ** i
        sizes.append((size, size, size))  
        if i > 7:
            sizes.append((size, size // 2, size * 2))  # Rectangular size 1
            sizes.append((size // 2, size, size * 2))  # Rectangular size 2
            sizes.append((size * 2, size // 2, size))  # Rectangular size 3

    sparsities = [0.1, 0.5, 0.9]
    data_dir = "data"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cublas', action='store_true', help="Run tests using the cuBLAS implementation")
    args = parser.parse_args()

    validate(sizes, sparsities, data_dir, use_cublas=args.use_cublas)

