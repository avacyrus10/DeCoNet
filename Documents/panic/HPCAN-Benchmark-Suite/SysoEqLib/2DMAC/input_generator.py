import numpy as np
import os

def generate_sparse_matrix(shape, sparsity):
    """Generates a sparse matrix of given shape and sparsity."""
    matrix = np.random.rand(*shape)  
    matrix[matrix > sparsity] = 0  
    return (matrix * 100).astype(np.int32)  

def save_data_to_file(data, filename):
    """Saves the matrix data to a binary file."""
    data.tofile(filename)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

sizes = []
for i in range(7, 17):
    size = 2**i
    sizes.append((size, size))
    if i > 7:
        sizes.append((size, 2**(i-1)))
        sizes.append((2**(i-1), size))

sparsities = [0.1, 0.5, 0.9]  # Sparsity levels

for widthA, heightA in sizes:
    for sparsity in sparsities:
        widthB = widthA

        input_a = generate_sparse_matrix((heightA, widthA), sparsity)
        input_b = generate_sparse_matrix((widthA, widthB), sparsity)

        input_file_a = f"{data_dir}/input_a_{heightA}x{widthA}_sparsity_{sparsity}.bin"
        input_file_b = f"{data_dir}/input_b_{widthA}x{widthB}_sparsity_{sparsity}.bin"

        # Save files
        save_data_to_file(input_a, input_file_a)
        save_data_to_file(input_b, input_file_b)

        # Verify files were created
        if not os.path.exists(input_file_a) or not os.path.exists(input_file_b):
            print(f"Error: Failed to create {input_file_a} or {input_file_b}")
            exit(1)

print(f"Input files generated successfully in {data_dir}")

