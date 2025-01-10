#include <cuda_runtime.h>
#include <iostream>

__global__ void reductionKernel(int* input, int* output, int n) {
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
}

void checkCudaError(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << file << ":" << line << " in " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)

void reduce(int* input, int* output, int n) {
    int* d_input;
    int* d_output;
    size_t inputSize = n * sizeof(int);
    size_t outputSize = ((n + 255) / 256) * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputSize));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reductionKernel<<<blocks, threads, threads * sizeof(int)>>>(d_input, d_output, n);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int* h_output = new int[blocks];
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));

    int total = 0;
    for (int i = 0; i < blocks; ++i) {
        total += h_output[i];
    }
    *output = total;

    delete[] h_output;
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}
