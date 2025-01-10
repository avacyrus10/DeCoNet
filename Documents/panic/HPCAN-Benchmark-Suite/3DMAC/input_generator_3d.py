import numpy as np
import os

def generate_3d_data(shape, sparsity):
    """Generates a sparse 3D matrix of given shape and sparsity."""
    matrix = np.random.rand(*shape)
    matrix[matrix > sparsity] = 0
    return (matrix * 100).astype(np.int32)  

def save_data_to_file(data, filename):
    """Saves the 3D matrix data to a binary file."""
    data.tofile(filename)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

sizes = []
for i in range(7, 17):  
    size = 2 ** i
    sizes.append((size, size, size))  
    if i > 7:
        sizes.append((size, size // 2, size * 2))  # Rectangular size 1
        sizes.append((size // 2, size, size * 2))  # Rectangular size 2
        sizes.append((size * 2, size // 2, size))  # Rectangular size 3

sparsities = [0.1, 0.5, 0.9]

for depth, height, width in sizes:
    for sparsity in sparsities:
        input_a = generate_3d_data((depth, height, width), sparsity)
        input_b = generate_3d_data((depth, height, width), sparsity)

        input_file_a = f"{data_dir}/input_a_{depth}x{height}x{width}_sparsity_{sparsity}.bin"
        input_file_b = f"{data_dir}/input_b_{depth}x{height}x{width}_sparsity_{sparsity}.bin"

        save_data_to_file(input_a, input_file_a)
        save_data_to_file(input_b, input_file_b)

print(f"3D input files generated in {data_dir}")

