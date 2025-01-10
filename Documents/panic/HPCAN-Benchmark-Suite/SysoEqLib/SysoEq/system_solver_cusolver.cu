#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

void solveWithCusolver(double* A, double* b, double* x, int size) {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    double* d_A;
    double* d_b;
    int* devInfo;
    int* pivotArray;
    int workspace_size = 0;
    double* workspace = nullptr;

    size_t matrixSize = size * size * sizeof(double);
    size_t vectorSize = size * sizeof(double);

    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_b, vectorSize);
    cudaMalloc((void**)&pivotArray, size * sizeof(int));
    cudaMalloc((void**)&devInfo, sizeof(int));

    cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vectorSize, cudaMemcpyHostToDevice);

    cusolverDnDgetrf_bufferSize(cusolverH, size, size, d_A, size, &workspace_size);
    cudaMalloc(&workspace, workspace_size * sizeof(double));

    cusolverDnDgetrf(cusolverH, size, size, d_A, size, workspace, pivotArray, devInfo);
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    if (info != 0) {
        std::cerr << "LU factorization failed with error code " << info << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(pivotArray);
        cudaFree(devInfo);
        cudaFree(workspace);
        cusolverDnDestroy(cusolverH);
        return;
    }

    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, size, 1, d_A, size, pivotArray, d_b, size, devInfo);
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    if (info != 0) {
        std::cerr << "Solve operation failed with error code " << info << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(pivotArray);
        cudaFree(devInfo);
        cudaFree(workspace);
        cusolverDnDestroy(cusolverH);
        return;
    }

    cudaMemcpy(x, d_b, vectorSize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(pivotArray);
    cudaFree(devInfo);
    cudaFree(workspace);

    cusolverDnDestroy(cusolverH);
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

    solveWithCusolver(A.data(), b.data(), x.data(), size);

    std::ofstream outputFile("solution.bin", std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(x.data()), size * sizeof(double));
    outputFile.close();

    return 0;
}

