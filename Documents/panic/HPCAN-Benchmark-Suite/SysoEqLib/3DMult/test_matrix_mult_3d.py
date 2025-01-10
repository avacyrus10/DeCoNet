import numpy as np
import os
import subprocess
import argparse


def generate_3d_array(shape, sparsity=0.0):
    """Generates a sparse 3D matrix of given shape and sparsity."""
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array


def save_3d_array(array, filename):
    """Saves a 3D matrix to a text file."""
    with open(filename, 'w') as f:
        for slice_ in array:
            np.savetxt(f, slice_)
            f.write("\n")


def load_3d_array(filename, shape):
    """Loads a 3D matrix from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    array = np.loadtxt(filename).reshape(shape)
    return array


def matrix_mult_3d(A, B):
    """Performs element-wise 3D matrix multiplication."""
    depth = A.shape[0]
    result = np.zeros((A.shape[0], A.shape[1], B.shape[2]), dtype=np.float32)
    for d in range(depth):
        result[d] = np.dot(A[d], B[d])
    return result


def generate_inputs(data_dir, sizes, sparsities):
    """Generates 3D matrix inputs."""
    os.makedirs(data_dir, exist_ok=True)
    for widthA, heightA, depthA in sizes:
        for sparsity in sparsities:
            widthB = widthA  

            matrixA = generate_3d_array((depthA, heightA, widthA), sparsity)
            matrixB = generate_3d_array((depthA, widthA, widthB), sparsity)

            matrixA_file = os.path.join(data_dir, f"matrixA_{widthA}x{heightA}x{depthA}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{widthA}x{widthB}x{depthA}_{sparsity}.txt")

            save_3d_array(matrixA, matrixA_file)
            save_3d_array(matrixB, matrixB_file)

            print(f"Generated: {matrixA_file}, {matrixB_file}")


def validate(data_dir, sizes, sparsities, use_cublas=False):
    """Validates 3D matrix multiplication using the specified implementation."""
    for widthA, heightA, depthA in sizes:
        for sparsity in sparsities:
            widthB = widthA

            matrixA_file = os.path.join(data_dir, f"matrixA_{widthA}x{heightA}x{depthA}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{widthA}x{widthB}x{depthA}_{sparsity}.txt")
            output_file = os.path.join(data_dir, f"output_{widthA}x{heightA}x{depthA}_{sparsity}.txt")

            # Choose the binary
            exec_bin = "./bin/matrixMult3D_cublas" if use_cublas else "./bin/matrixMult3D"

            # Run the CUDA program
            print(f"Running: {exec_bin} {matrixA_file} {matrixB_file} {output_file} {widthA} {heightA} {depthA} {widthB}")
            result = subprocess.run([exec_bin, matrixA_file, matrixB_file, output_file, str(widthA), str(heightA), str(depthA), str(widthB)])

            if result.returncode != 0:
                print(f"Error running {exec_bin} for size={widthA}x{heightA}x{depthA}, sparsity={sparsity}")
                continue

            try:
                output_data = load_3d_array(output_file, (depthA, heightA, widthB))
            except Exception as e:
                print(f"Error loading output file: {e}")
                continue

            matrixA = load_3d_array(matrixA_file, (depthA, heightA, widthA))
            matrixB = load_3d_array(matrixB_file, (depthA, widthA, widthB))
            expected_output = matrix_mult_3d(matrixA, matrixB)

            if np.allclose(output_data, expected_output, atol=1e-3):
                print(f"Test passed for size={widthA}x{heightA}x{depthA}, sparsity={sparsity}")
            else:
                print(f"Test failed for size={widthA}x{heightA}x{depthA}, sparsity={sparsity}")
                diff = np.abs(output_data - expected_output)
                max_diff = np.max(diff)
                max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"Max difference: {max_diff} at index {max_diff_indices}")
                print(f"CUDA Output at max diff: {output_data[max_diff_indices]}")
                print(f"Expected Output at max diff: {expected_output[max_diff_indices]}")
                print(f"Difference at max diff: {diff[max_diff_indices]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_only', action='store_true', help="Only generate inputs")
    parser.add_argument('--use_cublas', action='store_true', help="Use cuBLAS implementation")
    args = parser.parse_args()

    data_dir = "data"

    sizes = []
    for i in range(7, 17):  
        size = 2 ** i
        sizes.append((size, size, size))  # Cube size
        if i > 7:
            sizes.append((size, size // 2, size * 2))  # Rectangular size 1
            sizes.append((size // 2, size, size * 2))  # Rectangular size 2
            sizes.append((size * 2, size // 2, size))  # Rectangular size 3

    sparsities = [0.1, 0.5, 0.9]

    if args.generate_only:
        generate_inputs(data_dir, sizes, sparsities)
    else:
        validate(data_dir, sizes, sparsities, use_cublas=args.use_cublas)

