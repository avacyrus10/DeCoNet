import numpy as np
import subprocess
import os
import sys

def run_cuda_solver(executable, input_matrix_file, input_vector_file, size):
    output_file = "solution.bin"

    subprocess.run([executable, input_matrix_file, input_vector_file, str(size)])

    cuda_solution = np.fromfile(output_file, dtype=np.float64).reshape(size)
    return cuda_solution

def validate_solver(solver_type):
    matrices = [f for f in os.listdir('data') if 'matrix' in f and f.endswith('.bin')]
    vectors = [f for f in os.listdir('data') if 'vector' in f and f.endswith('.bin')]

    matrices.sort()
    vectors.sort()

    if solver_type == 'manual':
        executable = './system_solver'
    elif solver_type == 'lib':
        executable = './system_solver_cusolver'
    else:
        print("Unknown solver type. Use 'manual' or 'lib'.")
        return

    for matrix_file, vector_file in zip(matrices, vectors):
        size = int(matrix_file.split('_')[1].split('x')[0])

        input_matrix = np.fromfile(f'data/{matrix_file}', dtype=np.float64).reshape(size, size)
        input_vector = np.fromfile(f'data/{vector_file}', dtype=np.float64)

        cuda_solution = run_cuda_solver(executable, f'data/{matrix_file}', f'data/{vector_file}', size)

        expected_solution = np.linalg.solve(input_matrix, input_vector)

        if np.allclose(cuda_solution, expected_solution, atol=1e-3): 
            print(f"Test passed for system size {size}")
        else:
            print(f"Test failed for system size {size}")
            print(f"Max difference: {np.max(np.abs(cuda_solution - expected_solution))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_system_of_equations.py <manual|lib>")
        sys.exit(1)

    solver_type = sys.argv[1]
    validate_solver(solver_type)

