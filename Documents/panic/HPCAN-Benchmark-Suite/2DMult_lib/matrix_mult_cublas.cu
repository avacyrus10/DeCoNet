#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cublas_v2.h>

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
    if (argc != 7) {
        std::cerr << "Usage: ./matrixMult <matrixA> <matrixB> <output> <widthA> <heightA> <widthB>\n";
        return 1;
    }

    const char* matrixAFile = argv[1];
    const char* matrixBFile = argv[2];
    const char* outputFile = argv[3];
    int widthA = std::atoi(argv[4]);
    int heightA = std::atoi(argv[5]);
    int widthB = std::atoi(argv[6]);

    std::vector<float> hostA, hostB, hostC(heightA * widthB);
    loadMatrix(matrixAFile, hostA, heightA, widthA);
    loadMatrix(matrixBFile, hostB, widthA, widthB);

    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, widthA * heightA * sizeof(float));
    cudaMalloc((void**)&devB, widthA * widthB * sizeof(float));
    cudaMalloc((void**)&devC, heightA * widthB * sizeof(float));

    cudaMemcpy(devA, hostA.data(), widthA * heightA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, widthB, heightA, widthA, &alpha, devB, widthB, devA, widthA, &beta, devC, widthB);

    cudaMemcpy(hostC.data(), devC, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, heightA, widthB);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cublasDestroy(handle);

    return 0;
}

