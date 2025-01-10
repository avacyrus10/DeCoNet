#include <cuda_runtime.h>
#include <math.h>

__global__ void dct_kernel(const float* d_input, float* d_output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        double sum = 0.0; 
        for (int n = 0; n < N; ++n) {
            sum += d_input[n] * cos((M_PI / N) * (n + 0.5) * k);
        }
        double alpha = (k == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
        d_output[k] = (float)(alpha * sum);
    }
}

extern "C" void dct(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dct_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
}

__global__ void idct_kernel(const float* d_input, float* d_output, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        double sum = 0.0; 
        for (int k = 0; k < N; ++k) {
            double alpha = (k == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            sum += alpha * d_input[k] * cos((M_PI / N) * (n + 0.5) * k);
        }
        d_output[n] = (float)sum;
    }
}


extern "C" void idct(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    idct_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
}

