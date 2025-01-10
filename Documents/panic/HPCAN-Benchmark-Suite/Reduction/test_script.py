import numpy as np
import os
import subprocess

def generate_input_data(size):
    return np.random.randint(0, 100, size, dtype=np.int32)

def save_data_to_file(data, filename):
    data.tofile(filename)

def read_data_from_file(filename, dtype=np.int32):
    return np.fromfile(filename, dtype=dtype)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

sizes = [2**i for i in range(7, 10)] + [2**i + j for i in range(7, 10) for j in [1, 2, 4, 8, 16]]

def test_reduction(method="normal"):
    executable = "./reduce_program" if method == "normal" else "./cooperative_grid_reduce"
    
    for size in sizes:
        input_data = generate_input_data(size)
        input_file = f"{data_dir}/input_{size}.bin"
        output_file = f"{data_dir}/output_{size}.bin"

        save_data_to_file(input_data, input_file)

        subprocess.run(
            [executable, input_file, output_file, str(size), method],
            check=True
        )

        output_result = read_data_from_file(output_file, dtype=np.int32)
        expected_result = np.sum(input_data)

        assert output_result == expected_result, f"Test failed for size {size}: expected {expected_result}, got {output_result}"

        print(f"Test passed for size {size}")

if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "normal"
    test_reduction(method)

