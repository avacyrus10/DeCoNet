import numpy as np
import os
import subprocess
import argparse

def load_3d_array_from_file(filename, shape):
    try:
        data = np.loadtxt(filename, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error loading file {filename}: {e}")

    if data.size != np.prod(shape):
        raise ValueError(f"File {filename} contains {data.size} elements, but {np.prod(shape)} were expected.")
    
    return data.reshape(shape)

def convolve_3d(input, kernel):
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

def validate_outputs(use_cudnn):
    data_dir = "data"
    kernel_size = 3

    # Select binary based on the cudnn flag
    binary = "./bin/cudnn_convolve3D" if use_cudnn else "./bin/convolve3D"

    for filename in os.listdir(data_dir):
        if filename.startswith("input_"):
            input_file = os.path.join(data_dir, filename)
            kernel_file = os.path.join(data_dir, f"kernel_{kernel_size}.txt")
            base_name = filename.split('.')[0]
            dimensions = base_name.split('_')[1].split('x')
            width, height, depth = map(int, dimensions)
            sparsity = float(base_name.split('_')[2])
            output_file = os.path.join(data_dir, f"output_{width}x{height}x{depth}_{sparsity}.txt")

            print(f"Running binary: {binary} with inputs: {input_file}, {kernel_file}, {output_file}")
            subprocess.run([
                binary, input_file, kernel_file, output_file,
                str(width), str(height), str(depth), str(kernel_size), "0"
            ])

            output_shape = (depth - kernel_size + 1, height - kernel_size + 1, width - kernel_size + 1)
            output_data = load_3d_array_from_file(output_file, output_shape)
            input_data = load_3d_array_from_file(input_file, (depth, height, width))
            kernel_data = load_3d_array_from_file(kernel_file, (kernel_size, kernel_size, kernel_size))

            expected_output = convolve_3d(input_data, kernel_data)
            tolerance = 1e-3
            result = np.allclose(output_data, expected_output, atol=tolerance)
            print(f"Validation for {filename}: {'Passed' if result else 'Failed'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate 3D convolution outputs.")
    parser.add_argument("--cudnn", action="store_true", help="Use CUDNN-based binary for validation")
    args = parser.parse_args()

    validate_outputs(use_cudnn=args.cudnn)

