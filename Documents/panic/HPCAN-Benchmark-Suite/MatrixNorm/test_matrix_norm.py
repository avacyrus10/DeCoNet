import numpy as np
import subprocess
import os
import sys

def get_matrices():
    files = os.listdir('data')
    return ['data/' + f for f in files if f.endswith('.bin')]

def run_cuda_program(input_file, size, method):
    executable = "./matrix_norm" if method == "custom" else "./matrix_norm_cublas"
    result = subprocess.run([executable, input_file, str(size)], capture_output=True, text=True)
    output = result.stdout.strip()
    if "Computed norm: " not in output:
        return None
    norm_str = output.split("Computed norm: ")[1].strip()
    return float(norm_str)

def validate_matrices(method):
    matrices = get_matrices()
    for input_file in matrices:
        size = int(input_file.split('_')[1].split('x')[0])

        input_matrix = np.fromfile(input_file, dtype=np.float64).reshape(size, size)
        cuda_norm = run_cuda_program(input_file, size, method)

        if cuda_norm is None:
            print(f"Test failed for matrix size {size}x{size} using {method} method.")
            continue

        expected_norm = np.linalg.norm(input_matrix, 'fro')
        if np.allclose(cuda_norm, expected_norm, atol=1e-5):
            print(f"Test passed for matrix size {size}x{size} using {method} method.")
        else:
            print(f"Test failed for matrix size {size}x{size} using {method} method.")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["custom", "cublas"]:
        sys.exit(1)

    method = sys.argv[1]
    validate_matrices(method)
