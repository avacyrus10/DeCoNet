#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "spmv.h"

#define CUDA_CHECK(call)                                                        \
{                                                                               \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                  << cudaGetErrorString(err) << std::endl;                      \
        exit(1);                                                                \
    }                                                                           \
}

__global__ void spmv_kernel(const float* val, const int* rowPtr, const int* colIdx, const float* x, float* y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
            sum += val[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}

void spmv(const std::vector<float>& val, const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
          const std::vector<float>& x, std::vector<float>& y, int num_rows) {
    float *d_val, *d_x, *d_y;
    int *d_rowPtr, *d_colIdx;
    size_t val_size = val.size() * sizeof(float);
    size_t rowPtr_size = rowPtr.size() * sizeof(int);
    size_t colIdx_size = colIdx.size() * sizeof(int);
    size_t x_size = x.size() * sizeof(float);
    size_t y_size = y.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_val, val_size));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, rowPtr_size));
    CUDA_CHECK(cudaMalloc(&d_colIdx, colIdx_size));
    CUDA_CHECK(cudaMalloc(&d_x, x_size));
    CUDA_CHECK(cudaMalloc(&d_y, y_size));

    CUDA_CHECK(cudaMemcpy(d_val, val.data(), val_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), rowPtr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), colIdx_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    spmv_kernel<<<grid_size, block_size>>>(d_val, d_rowPtr, d_colIdx, d_x, d_y, num_rows);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, y_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

void spmv_cpu(const std::vector<float>& val, const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
              const std::vector<float>& x, std::vector<float>& y, int num_rows) {
    for (int i = 0; i < num_rows; ++i) {
        float sum = 0.0f;
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            sum += val[j] * x[colIdx[j]];
        }
        y[i] = sum;
    }
}

bool validate(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, float tolerance) {
    if (gpu_result.size() != cpu_result.size()) return false;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU " << gpu_result[i]
                      << ", CPU " << cpu_result[i] << std::endl;
            return false;
        }
    }

    return true;
}

