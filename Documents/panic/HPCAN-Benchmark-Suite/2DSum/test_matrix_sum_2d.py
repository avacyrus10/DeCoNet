import numpy as np
import subprocess
import os
import argparse

def generate_2d_array(shape, sparsity=0.0):
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array

def save_2d_array(array, filename):
    with open(filename, 'w') as f:
        np.savetxt(f, array)

def load_2d_array(filename, shape):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Output file {filename} not found.")
    array = np.loadtxt(filename).reshape(shape)
    return array

def matrix_sum_2d(A, B):
    return A + B

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def generate_inputs(sizes, sparsities):
    for width, height in sizes:
        for sparsity in sparsities:
            matrixA = generate_2d_array((height, width), sparsity)
            matrixB = generate_2d_array((height, width), sparsity)

            matrixA_file = os.path.join(data_dir, f"matrixA_{width}x{height}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{width}x{height}_{sparsity}.txt")

            save_2d_array(matrixA, matrixA_file)
            save_2d_array(matrixB, matrixB_file)

    print(f"Input files generated successfully in {data_dir}")

def run_tests(sizes, sparsities, use_cublas=False):
    for width, height in sizes:
        for sparsity in sparsities:
            matrixA_file = os.path.join(data_dir, f"matrixA_{width}x{height}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{width}x{height}_{sparsity}.txt")
            output_file = os.path.join(data_dir, f"output_{width}x{height}_{sparsity}.txt")

            if use_cublas:
                exec_bin = "./bin/matrixSum2D_cublas"
            else:
                exec_bin = "./bin/matrixSum2D"

            result = subprocess.run([exec_bin, matrixA_file, matrixB_file, output_file, str(width), str(height)])

            if result.returncode != 0:
                print(f"Error running command: {exec_bin}. Exiting with code {result.returncode}.")
                continue

            output_shape = (height, width)
            try:
                output_data = load_2d_array(output_file, output_shape)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                continue
            except ValueError as e:
                print(f"Error reshaping output for size={width}x{height}, sparsity={sparsity}: {e}")
                print(f"Expected shape: {output_shape}")
                continue

            matrixA = load_2d_array(matrixA_file, output_shape)
            matrixB = load_2d_array(matrixB_file, output_shape)

            expected_output = matrix_sum_2d(matrixA, matrixB)
            tolerance = 1e-3
            result = np.allclose(output_data, expected_output, atol=tolerance)
            status = "Passed" if result else "Failed"
            print(f"Test for size={width}x{height}, sparsity={sparsity}: {status}")
            if not result:
                diff = np.abs(output_data - expected_output)
                max_diff = np.max(diff)
                max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"Max difference: {max_diff} at index {max_diff_indices}")
                print(f"CUDA Output at max diff: {output_data[max_diff_indices]}")
                print(f"Expected Output at max diff: {expected_output[max_diff_indices]}")
                print(f"Difference at max diff: {diff[max_diff_indices]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cublas', action='store_true', help="Run tests using the cuBLAS implementation")
    parser.add_argument('--generate_only', action='store_true', help="Generate inputs only, do not run tests")
    args = parser.parse_args()

    sizes = []
    for i in range(7, 17):  
        size = 2 ** i
        sizes.append((size, size))  
        if i > 7:
            sizes.append((size, 2 ** (i - 1)))  # Rectangular size 1
            sizes.append((2 ** (i - 1), size))  # Rectangular size 2

    sparsities = [0.1, 0.5, 0.9]

    if args.generate_only:
        generate_inputs(sizes, sparsities)
    else:
        run_tests(sizes, sparsities, use_cublas=args.use_cublas)

