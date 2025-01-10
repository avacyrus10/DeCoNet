import numpy as np
import os

def generate_2d_array(shape, sparsity=0.0):
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array

def save_2d_array(array, filename):
    with open(filename, 'w') as f:
        np.savetxt(f, array)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def generate_inputs():
    sizes = [(2**i, 2**i) for i in range(7, 17)]
    sparsities = [0.1, 0.5, 0.9]
    kernel_size = 3  

    for width, height in sizes:
        for sparsity in sparsities:
            input_data = generate_2d_array((height, width), sparsity)
            kernel_data = generate_2d_array((kernel_size, kernel_size))

            input_file = os.path.join(data_dir, f"input_{width}x{height}_{sparsity}.txt")
            kernel_file = os.path.join(data_dir, f"kernel_{kernel_size}.txt")

            save_2d_array(input_data, input_file)
            save_2d_array(kernel_data, kernel_file)

if __name__ == "__main__":
    generate_inputs()

