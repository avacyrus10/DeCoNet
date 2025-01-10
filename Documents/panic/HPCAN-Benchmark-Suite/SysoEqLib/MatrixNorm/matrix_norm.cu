#include <cuda_runtime.h>
#include "matrix_norm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

__global__ void computeNormKernel(const double *matrix, double *norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        atomicAdd(norm, matrix[idx] * matrix[idx]);
    }
}

double computeMatrixNorm(const double *inputMatrix, int size) {
    double *d_matrix, *d_norm;
    double h_norm = 0.0;
    size_t matrixSize = size * size * sizeof(double);

    cudaMalloc((void **)&d_matrix, matrixSize);
    cudaMalloc((void **)&d_norm, sizeof(double));

    cudaMemcpy(d_matrix, inputMatrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm, &h_norm, sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((size * size + blockSize.x - 1) / blockSize.x);

    computeNormKernel<<<numBlocks, blockSize>>>(d_matrix, d_norm, size);

    cudaMemcpy(&h_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_norm);

    return sqrt(h_norm);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        return 1;
    }

    int size = std::stoi(argv[2]);

    std::vector<double> inputMatrix(size * size);

    std::ifstream inputFile(argv[1], std::ios::binary);
    inputFile.read(reinterpret_cast<char *>(inputMatrix.data()), size * size * sizeof(double));
    inputFile.close();

    double norm = computeMatrixNorm(inputMatrix.data(), size);

    std::cout << "Computed norm: " << norm << std::endl;

    return 0;
}

