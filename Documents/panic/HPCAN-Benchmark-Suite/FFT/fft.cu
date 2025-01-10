#include "fft.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__device__ int bit_reverse(int n, int bits) {
    int reversed = 0;
    for (int i = 0; i < bits; i++) {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    return reversed;
}

__global__ void bit_reversal(float2 *data, int N, int logN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int j = bit_reverse(i, logN);
        if (i < j) {
            float2 temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

__global__ void fft_kernel(float2 *data, int N, int step, int inverse) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int j = i / step * step * 2 + i % step;
        if (j + step < N) {
            float angle = (inverse ? 2.0f : -2.0f) * M_PI * (i % step) / (step * 2);
            float2 w = {cosf(angle), sinf(angle)};
            float2 u = data[j];
            float2 v = data[j + step];
            float2 t = {v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x};
            data[j] = {u.x + t.x, u.y + t.y};
            data[j + step] = {u.x - t.x, u.y - t.y};
        }
    }
}

void fft(float2 *data, int N) {
    int logN = log2(N);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    bit_reversal<<<num_blocks, threads_per_block>>>(data, N, logN);
    cudaDeviceSynchronize();

    for (int step = 1; step < N; step *= 2) {
        fft_kernel<<<num_blocks, threads_per_block>>>(data, N, step, 0);
        cudaDeviceSynchronize();
    }
}

void ifft(float2 *data, int N) {
    int logN = log2(N);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    bit_reversal<<<num_blocks, threads_per_block>>>(data, N, logN);
    cudaDeviceSynchronize();

    for (int step = 1; step < N; step *= 2) {
        fft_kernel<<<num_blocks, threads_per_block>>>(data, N, step, 1);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < N; i++) {
        data[i].x /= N;
        data[i].y /= N;
    }
}
