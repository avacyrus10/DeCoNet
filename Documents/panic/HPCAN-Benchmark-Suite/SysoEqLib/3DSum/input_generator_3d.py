import numpy as np
import os

def generate_3d_array(shape, sparsity=0.0):
    """Generates a sparse 3D matrix with specified dimensions and sparsity."""
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array

def save_3d_array(array, filename):
    """Saves a 3D matrix to a text file."""
    with open(filename, 'w') as f:
        for slice_ in array:
            np.savetxt(f, slice_)
            f.write("\n")

def generate_inputs(sizes, sparsities, data_dir):
    """Generates input files for testing."""
    os.makedirs(data_dir, exist_ok=True)
    for width, height, depth in sizes:
        for sparsity in sparsities:
            matrixA = generate_3d_array((depth, height, width), sparsity)
            matrixB = generate_3d_array((depth, height, width), sparsity)

            matrixA_file = os.path.join(data_dir, f"matrixA_{width}x{height}x{depth}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{width}x{height}x{depth}_{sparsity}.txt")

            save_3d_array(matrixA, matrixA_file)
            save_3d_array(matrixB, matrixB_file)

    print(f"Input files generated in {data_dir}")

if __name__ == "__main__":

    sizes = []
    for i in range(7, 17):  
        size = 2 ** i
        sizes.append((size, size, size))  
        if i > 7:
            sizes.append((size, size // 2, size * 2))  # Rectangular size 1
            sizes.append((size // 2, size, size * 2))  # Rectangular size 2
            sizes.append((size * 2, size // 2, size))  # Rectangular size 3

    sparsities = [0.1, 0.5, 0.9]
    data_dir = "data"

    generate_inputs(sizes, sparsities, data_dir)

