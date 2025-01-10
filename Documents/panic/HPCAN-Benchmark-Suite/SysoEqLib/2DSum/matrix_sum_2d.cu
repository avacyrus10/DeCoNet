#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

__global__ void matrixSum2D(float* A, float* B, float* C, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        C[row * width + col] = A[row * width + col] + B[row * width + col];
    }
}

void loadMatrix(const char* filename, std::vector<float>& matrix, int rows, int cols) {
    std::ifstream file(filename);
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        file >> matrix[i];
    }
}

void saveMatrix(const char* filename, const std::vector<float>& matrix, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * cols + j] << " ";
        }
        file << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Received " << argc << " arguments." << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "Arg " << i << ": " << argv[i] << std::endl;
    }

    if (argc != 6) {
        std::cerr << "Usage: ./matrixSum2D <matrixA> <matrixB> <output> <width> <height>\n";
        return 1;
    }

    const char* matrixAFile = argv[1];
    const char* matrixBFile = argv[2];
    const char* outputFile = argv[3];
    int width = std::atoi(argv[4]);
    int height = std::atoi(argv[5]);

    std::vector<float> hostA, hostB, hostC(width * height);
    loadMatrix(matrixAFile, hostA, height, width);
    loadMatrix(matrixBFile, hostB, height, width);

    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, width * height * sizeof(float));
    cudaMalloc((void**)&devB, width * height * sizeof(float));
    cudaMalloc((void**)&devC, width * height * sizeof(float));

    cudaMemcpy(devA, hostA.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
    matrixSum2D<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, width, height);

    cudaMemcpy(hostC.data(), devC, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, height, width);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

