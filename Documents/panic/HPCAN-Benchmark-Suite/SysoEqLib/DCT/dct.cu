#include "dct.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void dct_kernel(float *data, float *result, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += data[n] * cosf((M_PI / N) * (n + 0.5f) * k);
        }
        if (k == 0) {
            result[k] = sum / sqrtf(N);
        } else {
            result[k] = sum * sqrtf(2.0f / N);
        }
    }
}

__global__ void idct_kernel(float *data, float *result, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        float sum = data[0] / sqrtf(N);
        for (int k = 1; k < N; ++k) {
            sum += data[k] * cosf((M_PI / N) * (n + 0.5f) * k) * sqrtf(2.0f / N);
        }
        result[n] = sum;
    }
}

void dct(float *d_data, float *d_result, int N) {
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    dct_kernel<<<num_blocks, threads_per_block>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
}

void idct(float *d_data, float *d_result, int N) {
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    idct_kernel<<<num_blocks, threads_per_block>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
}

