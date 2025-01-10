#include <iostream>
#include <fstream>
#include <cuda.h>

__global__ void mac3DKernel(int *a, int *b, int *result, int width, int height, int depth) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int acc = 0;
        for (int d = 0; d < depth; ++d) {
            acc += a[d * height * width + row * width + col] * b[d * height * width + row * width + col];
        }
        result[row * width + col] = acc;
    }
}

void readData(const char* filename, int* data, int size) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }
    infile.read(reinterpret_cast<char*>(data), size * sizeof(int));
    infile.close();
}

void writeData(const char* filename, int* data, int size) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }
    outfile.write(reinterpret_cast<char*>(data), size * sizeof(int));
    outfile.close();
}

void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << message << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <input_a> <input_b> <output_result> <width> <height> <depth>" << std::endl;
        return 1;
    }

    std::cout << "Starting 3D MAC program..." << std::endl;

    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);
    int depth = std::stoi(argv[6]);

    int size = width * height * depth;
    int result_size = width * height;

    int* a, * b, * result;
    int *d_a, *d_b, *d_result;

    a = new int[size];
    b = new int[size];
    result = new int[result_size]();

    readData(argv[1], a, size);
    readData(argv[2], b, size);

    cudaMalloc(&d_a, size * sizeof(int));
    checkCudaError("cudaMalloc d_a");
    cudaMalloc(&d_b, size * sizeof(int));
    checkCudaError("cudaMalloc d_b");
    cudaMalloc(&d_result, result_size * sizeof(int));
    checkCudaError("cudaMalloc d_result");

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy d_a");
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy d_b");

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    mac3DKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, width, height, depth);
    checkCudaError("kernel launch");

    cudaMemcpy(result, d_result, result_size * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy d_result");

    writeData(argv[3], result, result_size);

    delete[] a;
    delete[] b;
    delete[] result;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    std::cout << "3D MAC operation completed successfully!" << std::endl;

    return 0;
}

