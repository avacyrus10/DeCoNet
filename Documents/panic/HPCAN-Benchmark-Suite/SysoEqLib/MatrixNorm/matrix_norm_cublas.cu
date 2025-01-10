#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

double computeMatrixNormCublas(const double *matrix, int size) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double result = 0.0;
    double *d_matrix;
    size_t matrixSize = size * size * sizeof(double);

    cudaMalloc(&d_matrix, matrixSize);
    cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);

    cublasDnrm2(handle, size * size, d_matrix, 1, &result);

    cudaFree(d_matrix);
    cublasDestroy(handle);

    return result;
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

    double norm = computeMatrixNormCublas(inputMatrix.data(), size);

    std::cout << "Computed norm: " << norm << std::endl;

    return 0;
}
