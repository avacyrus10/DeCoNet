
#include <iostream>
#include <fstream>
#include <cuda.h>

__global__ void matrixMultiplyKernel(int* a, int* b, int* result, int width_a, int height_a, int width_b) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height_a && col < width_b) {
        int value = 0;
        for (int i = 0; i < width_a; i++) {
            value += a[row * width_a + i] * b[i * width_b + col];
        }
        result[row * width_b + col] = value;
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

int main(int argc, char* argv[]) {
    if (argc != 7) { 
        std::cerr << "Usage: " << argv[0] << " <input_a> <input_b> <output_result> <width_a> <height_a> <width_b>" << std::endl;
        return 1;
    }

    std::cout << "Starting Matrix Multiply program..." << std::endl;

    int width_a = std::stoi(argv[4]);
    int height_a = std::stoi(argv[5]);
    int width_b = std::stoi(argv[6]);
    int height_b = width_a;  

    int size_a = width_a * height_a;
    int size_b = width_b * height_b;
    int size_result = height_a * width_b;

    int* a, * b, * result;
    int *d_a, *d_b, *d_result;

    a = new int[size_a];
    b = new int[size_b];
    result = new int[size_result]();

    readData(argv[1], a, size_a);
    readData(argv[2], b, size_b);

    cudaMalloc(&d_a, size_a * sizeof(int));
    cudaMalloc(&d_b, size_b * sizeof(int));
    cudaMalloc(&d_result, size_result * sizeof(int));

    cudaMemcpy(d_a, a, size_a * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);  
    dim3 gridSize((width_b + blockSize.x - 1) / blockSize.x, (height_a + blockSize.y - 1) / blockSize.y);
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, width_a, height_a, width_b);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpy(result, d_result, size_result * sizeof(int), cudaMemcpyDeviceToHost);

    writeData(argv[3], result, size_result);

    delete[] a;
    delete[] b;
    delete[] result;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    std::cout << "Matrix multiplication completed successfully!" << std::endl;

    return 0;
}

