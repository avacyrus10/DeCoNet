import numpy as np
import os
import sys
import subprocess

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
    array = np.loadtxt(filename).reshape(shape)
    return array

def generate_inputs(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    sizes = []
    for i in range(7, 17):
        size = 2 ** i
        sizes.append((size, size))
        if i > 7:
            sizes.append((size, 2 ** (i - 1)))
            sizes.append((2 ** (i - 1), size))
    sparsities = [0.1, 0.5, 0.9]
    kernel_size = 3

    for width, height in sizes:
        for sparsity in sparsities:
            input_shape = (height, width)
            input_data = generate_2d_array(input_shape, sparsity)
            kernel_shape = (kernel_size, kernel_size)
            kernel_data = generate_2d_array(kernel_shape)

            input_file = os.path.join(data_dir, f"input_{width}x{height}_{sparsity}.txt")
            kernel_file = os.path.join(data_dir, f"kernel_{kernel_size}.txt")

            save_2d_array(input_data, input_file)
            save_2d_array(kernel_data, kernel_file)

def run_tests(executable='./correlate2D', data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    sizes = []
    for i in range(7, 17):
        size = 2 ** i
        sizes.append((size, size))
        if i > 7:
            sizes.append((size, 2 ** (i - 1)))
            sizes.append((2 ** (i - 1), size))
    sparsities = [0.1, 0.5, 0.9]
    kernel_size = 3

    for width, height in sizes:
        for sparsity in sparsities:
            input_file = os.path.join(data_dir, f"input_{width}x{height}_{sparsity}.txt")
            kernel_file = os.path.join(data_dir, f"kernel_{kernel_size}.txt")
            output_file = os.path.join(data_dir, f"output_{width}x{height}_{sparsity}.txt")

            subprocess.run([executable, input_file, kernel_file, output_file, str(width), str(height), str(kernel_size)])

            output_shape = (height - kernel_size + 1, width - kernel_size + 1)
            try:
                output_data = load_2d_array(output_file, output_shape)
            except ValueError as e:
                print(f"Error reshaping output for size={width}x{height}, sparsity={sparsity}: {e}")
                continue

            input_data = load_2d_array(input_file, (height, width))
            kernel_data = load_2d_array(kernel_file, (kernel_size, kernel_size))
            expected_output = correlate_2d(input_data, kernel_data)
            tolerance = 1e-3
            result = np.allclose(output_data, expected_output, atol=tolerance)
            status = "Passed" if result else "Failed"
            print(f"Test for size={width}x{height}, sparsity={sparsity}: {status}")

def correlate_2d(input, kernel):
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)
    for y in range(output_height):
        for x in range(output_width):
            region = input[y:y+kernel_height, x:x+kernel_width]
            output[y, x] = np.sum(region * kernel)
    return output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_script.py [generate|test] [executable]")
        sys.exit(1)

    command = sys.argv[1]
    executable = sys.argv[2] if len(sys.argv) > 2 else './correlate2D'

    if command == "generate":
        generate_inputs()
    elif command == "test":
        run_tests(executable)
    else:
        print("Invalid command. Use 'generate' or 'test'.")

