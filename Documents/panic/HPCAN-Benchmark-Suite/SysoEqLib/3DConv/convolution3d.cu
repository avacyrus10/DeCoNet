#include <cuda_runtime.h>
#include "convolution3d.h"
#include <iostream>

__global__ void convolve3DKernel(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int outputWidth = width - kernelSize + 1;
    int outputHeight = height - kernelSize + 1;
    int outputDepth = depth - kernelSize + 1;

    if (x < outputWidth && y < outputHeight && z < outputDepth) {
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; ++k) {
            for (int j = 0; j < kernelSize; ++j) {
                for (int i = 0; i < kernelSize; ++i) {
                    int inputX = x + i;
                    int inputY = y + j;
                    int inputZ = z + k;
                    sum += input[(inputZ * height + inputY) * width + inputX] * kernel[(k * kernelSize + j) * kernelSize + i];
                }
            }
        }
        int outputIndex = (z * outputHeight + y) * outputWidth + x;
        output[outputIndex] = sum;
    }
}

void convolve3D(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    size_t inputSize = width * height * depth * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * kernelSize * sizeof(float);
    size_t outputSize = (width - kernelSize + 1) * (height - kernelSize + 1) * (depth - kernelSize + 1) * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width - kernelSize + 1 + blockSize.x - 1) / blockSize.x,
                  (height - kernelSize + 1 + blockSize.y - 1) / blockSize.y,
                  (depth - kernelSize + 1 + blockSize.z - 1) / blockSize.z);

    convolve3DKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, depth, kernelSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch convolve3DKernel: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

