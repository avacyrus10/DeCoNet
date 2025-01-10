#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void cooperativeReductionKernel(int* input, int* output, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];

    cg::grid_group grid = cg::this_grid();
    grid.sync();

    if (grid.thread_rank() == 0) {
        int total = 0;
        for (int i = 0; i < gridDim.x; ++i) {
            total += output[i];
        }
        output[0] = total;
    }
}

void reduceCooperative(int* input, int* output, int n) {
    int* d_input;
    int* d_output;
    size_t inputSize = n * sizeof(int);
    size_t outputSize = ((n + 255) / 256) * sizeof(int);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void* kernelArgs[] = {&d_input, &d_output, &n};

    cudaLaunchCooperativeKernel((void*)cooperativeReductionKernel, blocks, threads, kernelArgs, threads * sizeof(int));

    cudaDeviceSynchronize();

    int* h_output = new int[blocks];
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    *output = h_output[0];

    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

