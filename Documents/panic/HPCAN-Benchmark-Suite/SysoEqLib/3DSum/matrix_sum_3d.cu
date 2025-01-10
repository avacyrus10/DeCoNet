#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

__global__ void matrixSum3D(float* A, float* B, float* C, int width, int height, int depth) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < height && col < width && dep < depth) {
        C[(dep * height + row) * width + col] = A[(dep * height + row) * width + col] + B[(dep * height + row) * width + col];
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
    if (argc != 7) {
        std::cerr << "Usage: ./matrixSum3D <matrixA> <matrixB> <output> <width> <height> <depth>\n";
        return 1;
    }

    const char* matrixAFile = argv[1];
    const char* matrixBFile = argv[2];
    const char* outputFile = argv[3];
    int width = std::atoi(argv[4]);
    int height = std::atoi(argv[5]);
    int depth = std::atoi(argv[6]);

    std::vector<float> hostA, hostB, hostC(width * height * depth);
    loadMatrix(matrixAFile, hostA, height, width, depth);
    loadMatrix(matrixBFile, hostB, height, width, depth);

    float *devA, *devB, *devC;
    cudaMalloc((void**)&devA, width * height * depth * sizeof(float));
    cudaMalloc((void**)&devB, width * height * depth * sizeof(float));
    cudaMalloc((void**)&devC, width * height * depth * sizeof(float));

    cudaMemcpy(devA, hostA.data(), width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB.data(), width * height * depth * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    matrixSum3D<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, width, height, depth);

    cudaMemcpy(hostC.data(), devC, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);

    saveMatrix(outputFile, hostC, height, width, depth);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

