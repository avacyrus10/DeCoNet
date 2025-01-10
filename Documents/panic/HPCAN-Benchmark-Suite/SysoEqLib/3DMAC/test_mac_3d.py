import numpy as np
import os
import subprocess

def read_data_from_file(filename, dtype=np.int32):
    """Reads 3D matrix data from a binary file."""
    return np.fromfile(filename, dtype=dtype)

data_dir = "data"

def test_mac_3d():
    files = os.listdir(data_dir)
    input_files = [f for f in files if f.startswith("input_a")]
    for input_file_a in input_files:
        sparsity = input_file_a.split("_sparsity_")[-1].replace(".bin", "")
        input_file_b = input_file_a.replace("input_a", "input_b")
        output_file = input_file_a.replace("input_a", "output")

        # Extract dimensions
        depth, height, width = map(int, input_file_a.split("_")[2].split("x"))

        # Debugging print
        print(f"Running: ./mac_program {data_dir}/{input_file_a} {data_dir}/{input_file_b} {data_dir}/{output_file} {width} {height} {depth}")

        # Run the CUDA program
        subprocess.run(
            [
                "./mac_program",
                f"{data_dir}/{input_file_a}",
                f"{data_dir}/{input_file_b}",
                f"{data_dir}/{output_file}",
                str(width),
                str(height),
                str(depth),
            ],
            check=True
        )

        # Validate results
        input_a = read_data_from_file(f"{data_dir}/{input_file_a}").reshape((depth, height, width))
        input_b = read_data_from_file(f"{data_dir}/{input_file_b}").reshape((depth, height, width))
        output_result = read_data_from_file(f"{data_dir}/{output_file}").reshape((height, width))

        expected_result = np.sum(input_a * input_b, axis=0)

        assert np.allclose(output_result, expected_result), (
            f"Test failed for shape {(width, height, depth)} and sparsity {sparsity}: "
            f"expected \n{expected_result}, got \n{output_result}"
        )

        print(f"Test passed for shape {(width, height, depth)} with sparsity {sparsity}")

if __name__ == "__main__":
    test_mac_3d()

