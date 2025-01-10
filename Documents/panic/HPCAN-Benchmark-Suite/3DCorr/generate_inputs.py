import numpy as np
import os

def generate_3d_array(shape, sparsity=0.0):
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array

def save_3d_array(array, filename):
    with open(filename, 'w') as f:
        for slice in array:
            np.savetxt(f, slice)
            f.write('\n')

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def generate_inputs():

    sizes = []
    for i in range(7, 17):  
        size = 2 ** i
        sizes.append((size, size, size))  
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

            save_3d_array(input_data, input_file)
            save_3d_array(kernel_data, kernel_file)

    print(f"Input files generated successfully in {data_dir}")

if __name__ == "__main__":
    generate_inputs()

