#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cublas_v2.h>

void loadMatrix(const char* filename, std::vector<float>& matrix, int rows, int cols, int depth) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        exit(1);
    }

    matrix.resize(rows * cols * depth);
    for (int i = 0; i < rows * cols * depth; ++i) {
        if (!(file >> matrix[i])) {
            std::cerr << "Error: Insufficient data in file " << filename << "\n";
            exit(1);
        }
    }
}

void saveMatrix(const char* filename, const std::vector<float>& matrix, int rows, int cols, int depth) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        exit(1);
    }

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
    if (argc != 7) {
        std::cerr << "Usage: ./matrixSum3D_cublas <matrixA> <matrixB> <output> <width> <height> <depth>\n";
        return 1;
    }

    const char* matrixAFile = argv[1];
    const char* matrixBFile = argv[2];
    const char* outputFile = argv[3];
    int width = std::atoi(argv[4]);
    int height = std::atoi(argv[5]);
    int depth = std::atoi(argv[6]);

    if (width <= 0 || height <= 0 || depth <= 0) {
        std::cerr << "Error: Width, height, and depth must be positive integers.\n";
        return 1;
    }

    std::cout << "Parsed dimensions: width=" << width << ", height=" << height << ", depth=" << depth << "\n";

    std::vector<float> hostA, hostB, hostC(width * height * depth);
    loadMatrix(matrixAFile, hostA, height, width, depth);
    loadMatrix(matrixBFile, hostB, height, width, depth);

    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, width * height * depth * sizeof(float));
    cudaMalloc((void**)&devB, width * height * depth * sizeof(float));
    cudaMalloc((void**)&devC, width * height * depth * sizeof(float));

    cudaMemcpy(devA, hostA.data(), width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), width * height * depth * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: cuBLAS initialization failed.\n";
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
        return 1;
    }

    const float alpha = 1.0f;
    cublasSaxpy(handle, width * height * depth, &alpha, devA, 1, devB, 1);

    cudaMemcpy(hostC.data(), devB, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, height, width, depth);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cublasDestroy(handle);

    return 0;
}

