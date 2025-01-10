#include <cuda_runtime.h>
#include "matrix_inversion.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

__global__ void swapRows(double *matrix, double *invMatrix, int size, int row1, int row2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        double tmp = matrix[row1 * size + col];
        matrix[row1 * size + col] = matrix[row2 * size + col];
        matrix[row2 * size + col] = tmp;

        tmp = invMatrix[row1 * size + col];
        invMatrix[row1 * size + col] = invMatrix[row2 * size + col];
        invMatrix[row2 * size + col] = tmp;
    }
}

__global__ void normalizeRow(double *matrix, double *invMatrix, int size, int pivot) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        double diagVal = matrix[pivot * size + pivot];
        if (diagVal != 0) {
            matrix[pivot * size + col] /= diagVal;
            invMatrix[pivot * size + col] /= diagVal;
        }
    }
}

__global__ void eliminateRow(double *matrix, double *invMatrix, int size, int pivot) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && row != pivot) {
        double factor = matrix[row * size + pivot];
        for (int col = 0; col < size; ++col) {
            matrix[row * size + col] -= factor * matrix[pivot * size + col];
            invMatrix[row * size + col] -= factor * invMatrix[pivot * size + col];
        }
    }
}

void invertMatrix(const double *inputMatrix, double *outputMatrix, int size) {
    double *d_mat, *d_invMat;
    size_t matrixSize = size * size * sizeof(double);

    cudaMalloc((void **)&d_mat, matrixSize);
    cudaMalloc((void **)&d_invMat, matrixSize);

    cudaMemcpy(d_mat, inputMatrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_invMat, outputMatrix, matrixSize, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((size + blockSize.x - 1) / blockSize.x);

    for (int pivot = 0; pivot < size; ++pivot) {
        double maxVal = 0.0;
        int maxRow = pivot;
        for (int row = pivot; row < size; ++row) {
            double val = fabs(inputMatrix[row * size + pivot]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = row;
            }
        }

        if (maxRow != pivot) {
            swapRows<<<numBlocks, blockSize>>>(d_mat, d_invMat, size, pivot, maxRow);
            cudaDeviceSynchronize();
        }

        normalizeRow<<<numBlocks, blockSize>>>(d_mat, d_invMat, size, pivot);
        cudaDeviceSynchronize();

        eliminateRow<<<numBlocks, blockSize>>>(d_mat, d_invMat, size, pivot);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(outputMatrix, d_invMat, matrixSize, cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_invMat);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        return 1;
    }

    int size = std::stoi(argv[3]);

    std::vector<double> inputMatrix(size * size);
    std::vector<double> outputMatrix(size * size, 0);

    std::ifstream inputFile(argv[1], std::ios::binary);
    inputFile.read(reinterpret_cast<char *>(inputMatrix.data()), size * size * sizeof(double));
    inputFile.close();

    for (int i = 0; i < size; ++i) {
        outputMatrix[i * size + i] = 1.0;
    }

    invertMatrix(inputMatrix.data(), outputMatrix.data(), size);

    std::ofstream outputFile(argv[2], std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(outputMatrix.data()), size * size * sizeof(double));
    outputFile.close();

    return 0;
}
