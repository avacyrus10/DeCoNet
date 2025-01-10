import numpy as np
import os
import subprocess

def read_data_from_file(filename, dtype=np.int32):
    """Reads matrix data from a binary file."""
    return np.fromfile(filename, dtype=dtype)

data_dir = "data"

def test_matrix_multiply():
    files = os.listdir(data_dir)
    input_files = [f for f in files if f.startswith("input_a")]
    for input_file_a in input_files:
        sparsity = input_file_a.split("_sparsity_")[-1].replace(".bin", "")
        input_file_b = input_file_a.replace("input_a", "input_b")
        output_file = input_file_a.replace("input_a", "output")


        heightA, widthA = map(int, input_file_a.split("_")[2].split("x"))
        heightB, widthB = heightA, widthA  

        print(f"Running: ./mac_program {data_dir}/{input_file_a} {data_dir}/{input_file_b} {data_dir}/{output_file} {widthA} {heightA} {widthB}")

        subprocess.run(
            [
                "./mac_program",
                f"{data_dir}/{input_file_a}",
                f"{data_dir}/{input_file_b}",
                f"{data_dir}/{output_file}",
                str(widthA),
                str(heightA),
                str(widthB),
            ],
            check=True
        )

        input_a = read_data_from_file(f"{data_dir}/{input_file_a}").reshape((heightA, widthA))
        input_b = read_data_from_file(f"{data_dir}/{input_file_b}").reshape((widthA, widthB))
        output_result = read_data_from_file(f"{data_dir}/{output_file}").reshape((heightA, widthB))

        expected_result = np.dot(input_a, input_b)

        assert np.allclose(output_result, expected_result), (
            f"Test failed for shape {(heightA, widthA, widthB)} and sparsity {sparsity}: "
            f"expected \n{expected_result}, got \n{output_result}"
        )

        print(f"Test passed for shape {(heightA, widthA, widthB)} with sparsity {sparsity}")

if __name__ == "__main__":
    test_matrix_multiply()

