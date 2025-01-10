import numpy as np
import os
import subprocess
import sys

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_input_data(size):
    """Generate random integer data for input."""
    return np.random.randint(0, 100, size, dtype=np.int32)

def save_data_to_file(data, filename):
    """Save data to a binary file."""
    data.tofile(filename)

def read_data_from_file(filename, dtype=np.int32):
    """Read data from a binary file."""
    return np.fromfile(filename, dtype=dtype)

def generate_inputs():
    """Generate input files for testing."""
    sizes = [2**i for i in range(7, 10)] + [2**i + j for i in range(7, 10) for j in [1, 2, 4, 8, 16]]
    for size in sizes:
        input_data = generate_input_data(size)
        input_file = f"{DATA_DIR}/input_{size}.bin"
        save_data_to_file(input_data, input_file)
        print(f"Generated input file: {input_file}")

def run_tests():
    """Run tests for reduction."""
    sizes = [2**i for i in range(7, 10)] + [2**i + j for i in range(7, 10) for j in [1, 2, 4, 8, 16]]
    methods = ["normal", "cooperative"]
    executables = {
        "normal": "./reduce_program",
        "cooperative": "./cooperative_grid_reduce"
    }

    for method in methods:
        executable = executables[method]
        for size in sizes:
            input_file = f"{DATA_DIR}/input_{size}.bin"
            output_file = f"{DATA_DIR}/output_{size}_{method}.bin"

            subprocess.run(
                [executable, input_file, output_file, str(size), method],
                check=True
            )

            input_data = read_data_from_file(input_file)
            output_result = read_data_from_file(output_file, dtype=np.int32)
            expected_result = np.sum(input_data)

            assert output_result == expected_result, (
                f"Test failed for size {size} (method: {method}): "
                f"expected {expected_result}, got {output_result}"
            )

            print(f"Test passed for size {size} using method '{method}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_reduction.py <generate|test>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        generate_inputs()
    elif command == "test":
        run_tests()
    else:
        print("Invalid command. Use 'generate' or 'test'.")

