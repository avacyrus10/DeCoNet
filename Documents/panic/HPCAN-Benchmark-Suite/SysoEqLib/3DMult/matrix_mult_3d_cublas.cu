#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cublas_v2.h>

void loadMatrix(const char* filename, std::vector<float>& matrix, int rows, int cols, int depth) {
    std::ifstream file(filename);
    matrix.resize(rows * cols * depth);
    for (int i = 0; i < rows * cols * depth; ++i) {
        file >> matrix[i];
    }
}

void saveMatrix(const char* filename, const std::vector<float>& matrix, int rows, int cols, int depth) {
    std::ofstream file(filename);
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << matrix[(d * rows + i) * cols + j] << " ";
            }
            file << "\n";
        }
        file << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: ./matrixMult3D_cublas <matrixA> <matrixB> <output> <widthA> <heightA> <depthA> <widthB>\n";
        return 1;
    }

    const char* matrixAFile = argv[1];
    const char* matrixBFile = argv[2];
    const char* outputFile = argv[3];
    int widthA = std::atoi(argv[4]);
    int heightA = std::atoi(argv[5]);
    int depthA = std::atoi(argv[6]);
    int widthB = std::atoi(argv[7]);

    std::vector<float> hostA, hostB, hostC(heightA * widthB * depthA);
    loadMatrix(matrixAFile, hostA, heightA, widthA, depthA);
    loadMatrix(matrixBFile, hostB, widthA, widthB, depthA);

    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, widthA * heightA * depthA * sizeof(float));
    cudaMalloc((void**)&devB, widthA * widthB * depthA * sizeof(float));
    cudaMalloc((void**)&devC, heightA * widthB * depthA * sizeof(float));

    cudaMemcpy(devA, hostA.data(), widthA * heightA * depthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), widthA * widthB * depthA * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int d = 0; d < depthA; ++d) {
        float* sliceA = devA + d * widthA * heightA;
        float* sliceB = devB + d * widthA * widthB;
        float* sliceC = devC + d * heightA * widthB;

        // cuBLAS matrix multiplication: C = alpha * A * B + beta * C
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, widthB, heightA, widthA, &alpha, sliceB, widthB, sliceA, widthA, &beta, sliceC, widthB);
    }


    cudaMemcpy(hostC.data(), devC, heightA * widthB * depthA * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, heightA, widthB, depthA);


    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cublasDestroy(handle);

    return 0;
}

