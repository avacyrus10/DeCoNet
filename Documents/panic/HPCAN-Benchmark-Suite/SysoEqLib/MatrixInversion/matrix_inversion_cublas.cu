#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

void invertMatrixCublas(const double *inputMatrix, double *outputMatrix, int size) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double **d_Aarray, **d_Carray;
    int *d_pivotArray, *d_infoArray;
    int batchSize = 1;

    cudaMalloc(&d_Aarray, batchSize * sizeof(double *));
    cudaMalloc(&d_Carray, batchSize * sizeof(double *));
    cudaMalloc(&d_pivotArray, size * batchSize * sizeof(int));
    cudaMalloc(&d_infoArray, batchSize * sizeof(int));

    double *d_A, *d_C;
    cudaMalloc(&d_A, size * size * sizeof(double));
    cudaMalloc(&d_C, size * size * sizeof(double));

    cudaMemcpy(d_A, inputMatrix, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, outputMatrix, size * size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Aarray, &d_A, batchSize * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, &d_C, batchSize * sizeof(double *), cudaMemcpyHostToDevice);

    cublasDgetrfBatched(handle, size, d_Aarray, size, d_pivotArray, d_infoArray, batchSize);
    cublasDgetriBatched(handle, size, (const double **)d_Aarray, size, d_pivotArray, d_Carray, size, d_infoArray, batchSize);

    cudaMemcpy(outputMatrix, d_C, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_Aarray);
    cudaFree(d_Carray);
    cudaFree(d_pivotArray);
    cudaFree(d_infoArray);
    cublasDestroy(handle);
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

    invertMatrixCublas(inputMatrix.data(), outputMatrix.data(), size);

    std::ofstream outputFile(argv[2], std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(outputMatrix.data()), size * size * sizeof(double));
    outputFile.close();

    return 0;
}
