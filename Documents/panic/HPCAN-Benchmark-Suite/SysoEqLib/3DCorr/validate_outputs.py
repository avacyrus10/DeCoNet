import numpy as np
import os
import subprocess

def load_3d_array(filename, shape):
    data = np.loadtxt(filename, dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"File {filename} contains {data.size} elements, but {np.prod(shape)} were expected.")
    return data.reshape(shape)

def correlate_3d(input, kernel):
    input_depth, input_height, input_width = input.shape
    kernel_depth, kernel_height, kernel_width = kernel.shape
    output_depth = input_depth - kernel_depth + 1
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = np.zeros((output_depth, output_height, output_width), dtype=np.float32)
    for z in range(output_depth):
        for y in range(output_height):
            for x in range(output_width):
                region = input[z:z+kernel_depth, y:y+kernel_height, x:x+kernel_width]
                output[z, y, x] = np.sum(region * kernel)
    return output

def validate_outputs():
    data_dir = "data"
    kernel_size = 3
    binary = "./bin/correlate3D"

    for filename in os.listdir(data_dir):
        if filename.startswith("input_"):
            input_file = os.path.join(data_dir, filename)
            kernel_file = os.path.join(data_dir, f"kernel_{kernel_size}.txt")

            try:
                base_name = filename.split('.')[0]
                dimensions = base_name.split('_')[1].split('x')
                width, height, depth = map(int, dimensions[:3])
                sparsity = float(base_name.split('_')[2])
            except (IndexError, ValueError):
                print(f"Skipping invalid filename format: {filename}")
                continue

            output_file = os.path.join(data_dir, f"output_{width}x{height}x{depth}_{sparsity}.txt")

            subprocess.run([
                binary, input_file, kernel_file, output_file,
                str(width), str(height), str(depth), str(kernel_size)
            ])

            output_shape = (depth - kernel_size + 1, height - kernel_size + 1, width - kernel_size + 1)
            output_data = load_3d_array(output_file, output_shape)
            input_data = load_3d_array(input_file, (depth, height, width))
            kernel_data = load_3d_array(kernel_file, (kernel_size, kernel_size, kernel_size))

            expected_output = correlate_3d(input_data, kernel_data)
            result = np.allclose(output_data, expected_output, atol=1e-3)
            print(f"Validation for {filename}: {'Passed' if result else 'Failed'}")

if __name__ == "__main__":
    validate_outputs()

