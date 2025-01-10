#include <cuda_runtime.h>
#include <iostream>
#include "correlation2d.h"

__global__ void correlate2DKernel(const float* input, const float* kernel, float* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outputWidth = width - kernelSize + 1;
    int outputHeight = height - kernelSize + 1;

    if (x < outputWidth && y < outputHeight) {
        float sum = 0.0f;
        for (int j = 0; j < kernelSize; ++j) {
            for (int i = 0; i < kernelSize; ++i) {
                int inputX = x + i;
                int inputY = y + j;
                sum += input[inputY * width + inputX] * kernel[j * kernelSize + i];
            }
        }
        int outputIndex = y * outputWidth + x;
        output[outputIndex] = sum;
    }
}

void correlate2D(const float* input, const float* kernel, float* output, int width, int height, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    size_t inputSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);
    size_t outputSize = (width - kernelSize + 1) * (height - kernelSize + 1) * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width - kernelSize + 1 + blockSize.x - 1) / blockSize.x,
                  (height - kernelSize + 1 + blockSize.y - 1) / blockSize.y);

    correlate2DKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, kernelSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch correlate2DKernel: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
