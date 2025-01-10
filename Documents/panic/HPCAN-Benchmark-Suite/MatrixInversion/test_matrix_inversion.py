import numpy as np
import subprocess
import os
import sys

def get_input():
    files = os.listdir('data')
    bin_files = []

    for f in files:
        if f.endswith('.bin'):
            bin_files.append('data/' + f)

    return bin_files

def run_cuda_program(input_file, size, method):
    output_file = input_file.replace('input', 'output')
    executable = "./matrix_inversion" if method == "custom" else "./matrix_inversion_cublas"
    
    subprocess.run([executable, input_file, output_file, str(size), str(size)])

    output_matrix = np.fromfile(output_file, dtype=np.float64).reshape(size, size)
    return output_matrix

def validate(method):
    matrices = get_input()
    for input_file in matrices:
        size = int(input_file.split('_')[1].split('x')[0])

        input_matrix = np.fromfile(input_file, dtype=np.float64).reshape(size, size)
        cuda_output = run_cuda_program(input_file, size, method)

        expected_output = np.linalg.inv(input_matrix)
        if np.allclose(cuda_output, expected_output, atol=1e-5):  
            print(f"Test passed for matrix size {size}x{size} using {method} method.")
        else:
            max_diff = np.max(np.abs(cuda_output - expected_output))
            print(f"Test failed for matrix size {size}x{size} using {method} method.")
            print(f"Maximum difference: {max_diff}")
            print("Expected output:")
            print(expected_output)
            print("CUDA output:")
            print(cuda_output)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["custom", "cublas"]:
        print("Usage: python test_matrix_inversion.py [custom|cublas]")
        sys.exit(1)

    method = sys.argv[1]
    validate(method)
