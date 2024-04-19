#include "fft_cuda.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/swap.h>

const double PI = 3.14159265358979323846;

// CUDA kernel to perform FFT
__global__ void fft_kernel(thrust::complex<double>* a, int n, bool invert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int lg_n = 0;
    while ((1 << lg_n) < n)
        lg_n++;

    int j = 0;
    for (int i = 0; i < n; i++) {
        j = 0;
        for (int b = 0; b < lg_n; b++) {
            if (i & (1 << b))
                j |= 1 << (lg_n - 1 - b);
        }
        if (i < j)
            thrust::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        thrust::complex<double> wlen(cos(ang), sin(ang));
        for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
            thrust::complex<double> w(1);
            for (int j = 0; j < len / 2; j++) {
                thrust::complex<double> u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
            a[i] /= n;
        }
    }
}

// Wrapper function to call the CUDA kernel for FFT
void fft_cuda(std::vector<thrust::complex<double>>& a, bool invert) {
    int n = a.size();
    thrust::complex<double>* d_a;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, n * sizeof(thrust::complex<double>));

    // Copy input data from host to device
    cudaMemcpy(d_a, a.data(), n * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256; 
    int num_blocks = (n + block_size - 1) / block_size;
    fft_kernel<<<num_blocks, block_size>>>(d_a, n, invert);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(a.data(), d_a, n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
}
