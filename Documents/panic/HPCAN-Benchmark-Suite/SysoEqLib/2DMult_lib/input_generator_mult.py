import numpy as np
import os

def generate_2d_array(shape, sparsity=0.0):
    """Generates a 2D matrix with specified dimensions and sparsity."""
    array = np.random.rand(*shape).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(*shape) < sparsity
        array[mask] = 0
    return array

def save_2d_array(array, filename):
    """Saves a 2D matrix to a text file."""
    with open(filename, 'w') as f:
        np.savetxt(f, array)

def generate_inputs(data_dir, sizes, sparsities):
    """Generates input files for matrix multiplication."""
    os.makedirs(data_dir, exist_ok=True)
    for widthA, heightA in sizes:
        for sparsity in sparsities:
            # Randomly select non-square dimensions for B
            widthB = widthA if np.random.rand() > 0.5 else 2**np.random.randint(7, 16)

            matrixA = generate_2d_array((heightA, widthA), sparsity)
            matrixB = generate_2d_array((widthA, widthB))

            matrixA_file = os.path.join(data_dir, f"matrixA_{widthA}x{heightA}_{sparsity}.txt")
            matrixB_file = os.path.join(data_dir, f"matrixB_{widthA}x{widthB}_{sparsity}.txt")

            save_2d_array(matrixA, matrixA_file)
            save_2d_array(matrixB, matrixB_file)

            print(f"Generated: {matrixA_file}, {matrixB_file}")

    print(f"Inputs generated in {data_dir}")

if __name__ == "__main__":
    data_dir = "data"
    sizes = [(2**i, 2**i) for i in range(7, 17)]  # Square matrices
    sizes += [(2**i, 2**(i-1)) for i in range(8, 17)]  # Rectangular matrices
    sparsities = [0.1, 0.5, 0.9]  
    generate_inputs(data_dir, sizes, sparsities)

