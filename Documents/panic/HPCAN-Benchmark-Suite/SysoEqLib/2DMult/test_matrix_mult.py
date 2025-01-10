import numpy as np
import subprocess
import os

def load_2d_array(filename, shape):
    """Loads a 2D matrix from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    return np.loadtxt(filename).reshape(shape)

def matrix_mult(A, B):
    """Computes the matrix multiplication of A and B."""
    return np.dot(A, B)

def validate(data_dir, sizes, sparsities):
    """Validates the CUDA implementation."""
    for widthA, heightA in sizes:
        for sparsity in sparsities:
            widthB = widthA if np.random.rand() > 0.5 else 2**np.random.randint(7, 16)

            matrixA_file = os.path.join(data_dir, f"matrixA_{widthA}x{heightA}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{widthA}x{widthB}_{sparsity}.txt")
            output_file = os.path.join(data_dir, f"output_{widthA}x{heightA}_{sparsity}.txt")

            result = subprocess.run(["./bin/matrixMult", matrixA_file, matrixB_file, output_file, str(widthA), str(heightA), str(widthB)])

            if result.returncode != 0:
                print(f"Error running CUDA program for size={widthA}x{heightA}, sparsity={sparsity}")
                continue

            try:
                output_data = load_2d_array(output_file, (heightA, widthB))
            except Exception as e:
                print(f"Error loading output file: {e}")
                continue

            matrixA = load_2d_array(matrixA_file, (heightA, widthA))
            matrixB = load_2d_array(matrixB_file, (widthA, widthB))
            expected_output = matrix_mult(matrixA, matrixB)

            if np.allclose(output_data, expected_output, atol=1e-3):
                print(f"Test passed for size={widthA}x{heightA}, sparsity={sparsity}")
            else:
                print(f"Test failed for size={widthA}x{heightA}, sparsity={sparsity}")

if __name__ == "__main__":
    data_dir = "data"
    sizes = [(2**i, 2**i) for i in range(7, 17)]
    sizes += [(2**i, 2**(i-1)) for i in range(8, 17)]
    sparsities = [0.1, 0.5, 0.9]
    validate(data_dir, sizes, sparsities)

