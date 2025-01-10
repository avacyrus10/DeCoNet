#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

__global__ void matrixMul3D(float* A, float* B, float* C, int widthA, int heightA, int depthA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < heightA && col < widthB && depth < depthA) {
        float value = 0;
        for (int i = 0; i < widthA; ++i) {
            value += A[(depth * heightA + row) * widthA + i] * B[(depth * widthA + i) * widthB + col];
        }
        C[(depth * heightA + row) * widthB + col] = value;
    }
}

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
        std::cerr << "Usage: ./matrixMult3D <matrixA> <matrixB> <output> <widthA> <heightA> <depthA> <widthB>\n";
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

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((widthB + 15) / 16, (heightA + 15) / 16, depthA);
    matrixMul3D<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, widthA, heightA, depthA, widthB);

    cudaMemcpy(hostC.data(), devC, heightA * widthB * depthA * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, heightA, widthB, depthA);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

