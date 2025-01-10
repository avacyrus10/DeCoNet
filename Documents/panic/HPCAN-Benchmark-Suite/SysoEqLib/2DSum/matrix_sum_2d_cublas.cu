#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cublas_v2.h>

void loadMatrix(const char* filename, std::vector<float>& matrix, int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        exit(1);
    }
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        if (!(file >> matrix[i])) {
            std::cerr << "Error: Insufficient data in file " << filename << "\n";
            exit(1);
        }
    }
}

void saveMatrix(const char* filename, const std::vector<float>& matrix, int rows, int cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        exit(1);
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * cols + j] << " ";
        }
        file << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) { 
        std::cerr << "Usage: ./matrixSum2D_cublas <matrixA> <matrixB> <output> <width> <height>\n";
        return 1;
    }

    const char* matrixAFile = argv[1];
    const char* matrixBFile = argv[2];
    const char* outputFile = argv[3];
    int width = std::atoi(argv[4]);
    int height = std::atoi(argv[5]);

    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Width and height must be positive integers.\n";
        return 1;
    }

    std::vector<float> hostA, hostB, hostC(width * height);
    loadMatrix(matrixAFile, hostA, height, width);
    loadMatrix(matrixBFile, hostB, height, width);

    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, width * height * sizeof(float));
    cudaMalloc((void**)&devB, width * height * sizeof(float));
    cudaMalloc((void**)&devC, width * height * sizeof(float));

    cudaMemcpy(devA, hostA.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: cuBLAS initialization failed.\n";
        return 1;
    }

    const float alpha = 1.0f;
    cublasSaxpy(handle, width * height, &alpha, devA, 1, devB, 1);

    cudaMemcpy(hostC.data(), devB, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, height, width);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cublasDestroy(handle);

    return 0;
}

