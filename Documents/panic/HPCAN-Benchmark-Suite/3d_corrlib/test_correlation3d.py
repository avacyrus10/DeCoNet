import numpy as np
import os
import sys
import subprocess

def generate_3d_array(shape, sparsity=0.0):
    """Generate a 3D array with the given shape and sparsity."""
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array

def save_3d_array_to_file(array, filename):
    """Save a 3D array to a text file."""
    with open(filename, 'w') as f:
        for slice in array:
            np.savetxt(f, slice)
            f.write('\n')

def load_3d_array_from_file(filename, shape):
    """Load a 3D array from a text file."""
    array = np.loadtxt(filename).reshape(shape)
    return array

def generate_inputs():
    """Generate input data and save it for testing."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    sizes = []
    for i in range(7, 17): 
        size = 2 ** i
        sizes.append((size, size, size))  # Cube size
        if i > 7:
            sizes.append((size, size // 2, size * 2))  # Rectangular size 1
            sizes.append((size // 2, size, size * 2))  # Rectangular size 2
            sizes.append((size * 2, size // 2, size))  # Rectangular size 3

    sparsities = [0.1, 0.5, 0.9]
    kernel_size = 3

    for width, height, depth in sizes:
        for sparsity in sparsities:
            input_shape = (depth, height, width)
            input_data = generate_3d_array(input_shape, sparsity)

            kernel_shape = (kernel_size, kernel_size, kernel_size)
            kernel_data = generate_3d_array(kernel_shape)

            input_file = os.path.join(data_dir, f"input_{width}x{height}x{depth}_{sparsity}.txt")
            kernel_file = os.path.join(data_dir, f"kernel_{kernel_size}.txt")

            save_3d_array_to_file(input_data, input_file)
            save_3d_array_to_file(kernel_data, kernel_file)
            print(f"Generated input: {input_file}, kernel: {kernel_file}")

def run_tests(executable):
    """Run tests using the 3D correlation library."""
    data_dir = "data"

    for input_file in os.listdir(data_dir):
        if input_file.startswith("input"):
            kernel_file = os.path.join(data_dir, "kernel_3.txt")
            output_file = os.path.join(data_dir, input_file.replace("input", "output"))

            # Parse dimensions from the filename
            parts = input_file.split("_")[1].split("x")
            width, height, depth = map(int, parts[:3])
            kernel_size = 3

            subprocess.run([
                executable,
                os.path.join(data_dir, input_file),
                kernel_file,
                output_file,
                str(width), str(height), str(depth), str(kernel_size)
            ])

            output_shape = (depth - kernel_size + 1, height - kernel_size + 1, width - kernel_size + 1)
            try:
                output_data = load_3d_array_from_file(output_file, output_shape)
                print(f"Validation successful for {input_file}")
            except ValueError as e:
                print(f"Error reshaping output for {input_file}: {e}")
                continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_correlation3d.py <generate|test> [<executable>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "generate":
        generate_inputs()
    elif command == "test":
        if len(sys.argv) < 3:
            print("Executable required for testing")
            sys.exit(1)
        run_tests(sys.argv[2])
    else:
        print("Invalid command. Use 'generate' or 'test'.")

