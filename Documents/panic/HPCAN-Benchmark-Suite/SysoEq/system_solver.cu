#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

__global__ void swapRows(double* A, double* b, int size, int row1, int row2) {
    if (row1 != row2) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < size) {
            double temp = A[row1 * size + col];
            A[row1 * size + col] = A[row2 * size + col];
            A[row2 * size + col] = temp;
        }
        if (col == 0) { 
            double tempB = b[row1];
            b[row1] = b[row2];
            b[row2] = tempB;
        }
    }
}

__global__ void forwardElimination(double* A, double* b, int size, int pivot) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > pivot && row < size) {
        double factor = A[row * size + pivot] / A[pivot * size + pivot];
        for (int col = pivot; col < size; ++col) {
            A[row * size + col] -= factor * A[pivot * size + col];
        }
        b[row] -= factor * b[pivot];
    }
}

__global__ void backSubstitution(double* A, double* b, double* x, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size) {
        double sum = 0.0;
        for (int j = row + 1; j < size; ++j) {
            sum += A[row * size + j] * x[j];
        }
        x[row] = (b[row] - sum) / A[row * size + row];
    }
}

void findPivot(double* A, int size, int pivot, int* maxRow) {
    double maxValue = fabs(A[pivot * size + pivot]);
    *maxRow = pivot;
    for (int row = pivot + 1; row < size; ++row) {
        double value = fabs(A[row * size + pivot]);
        if (value > maxValue) {
            maxValue = value;
            *maxRow = row;
        }
    }
}

void solveSystem(double* A, double* b, double* x, int size) {
    double *d_A, *d_b, *d_x;
    size_t matrixSize = size * size * sizeof(double);
    size_t vectorSize = size * sizeof(double);

    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_b, vectorSize);
    cudaMalloc((void**)&d_x, vectorSize);

    cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vectorSize, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((size + blockSize.x - 1) / blockSize.x);

    for (int pivot = 0; pivot < size - 1; ++pivot) {
        int maxRow;
        findPivot(A, size, pivot, &maxRow);

        swapRows<<<numBlocks, blockSize>>>(d_A, d_b, size, pivot, maxRow);
        cudaDeviceSynchronize();


        forwardElimination<<<numBlocks, blockSize>>>(d_A, d_b, size, pivot);
        cudaDeviceSynchronize();
    }

    for (int row = size - 1; row >= 0; --row) {
        backSubstitution<<<1, 1>>>(d_A, d_b, d_x, size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(x, d_x, vectorSize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input matrix> <input vector> <size>" << std::endl;
        return 1;
    }

    int size = std::stoi(argv[3]);

    std::vector<double> A(size * size);
    std::vector<double> b(size);
    std::vector<double> x(size, 0);

    std::ifstream inputFileA(argv[1], std::ios::binary);
    inputFileA.read(reinterpret_cast<char*>(A.data()), size * size * sizeof(double));
    inputFileA.close();

    std::ifstream inputFileB(argv[2], std::ios::binary);
    inputFileB.read(reinterpret_cast<char*>(b.data()), size * sizeof(double));
    inputFileB.close();

    solveSystem(A.data(), b.data(), x.data(), size);

    std::ofstream outputFile("solution.bin", std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(x.data()), size * sizeof(double));
    outputFile.close();

    return 0;
}

