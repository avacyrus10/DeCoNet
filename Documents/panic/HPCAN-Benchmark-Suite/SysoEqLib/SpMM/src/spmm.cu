#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "spmm.h"

__global__ void spmmKernel(const float* __restrict__ d_val, const int* __restrict__ d_rowPtr,
                           const int* __restrict__ d_colIdx, const float* __restrict__ d_B, float* d_C,
                           int num_rows, int num_cols_B) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int j = 0; j < num_cols_B; ++j) {
            float sum = 0.0f;
            for (int idx = d_rowPtr[row]; idx < d_rowPtr[row + 1]; ++idx) {
                sum += d_val[idx] * d_B[d_colIdx[idx] * num_cols_B + j];
            }
            d_C[row * num_cols_B + j] = sum;
        }
    }
}

void spmm(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr, const std::vector<int> &h_colIdx,
          const std::vector<float> &h_B, std::vector<float> &h_C, int num_rows, int num_cols, int num_cols_B) {
    float *d_val, *d_B, *d_C;
    int *d_rowPtr, *d_colIdx;

    cudaMalloc(&d_val, h_val.size() * sizeof(float));
    cudaMalloc(&d_rowPtr, h_rowPtr.size() * sizeof(int));
    cudaMalloc(&d_colIdx, h_colIdx.size() * sizeof(int));
    cudaMalloc(&d_B, h_B.size() * sizeof(float));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_val, h_val.data(), h_val.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, h_rowPtr.data(), h_rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    spmmKernel<<<numBlocks, blockSize>>>(d_val, d_rowPtr, d_colIdx, d_B, d_C, num_rows, num_cols_B);

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_val);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_B);
    cudaFree(d_C);
}

void spmm_cpu(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr,
              const std::vector<int> &h_colIdx, const std::vector<float> &h_B,
              std::vector<float> &h_C, int num_rows, int num_cols_B) {
    for (int row = 0; row < num_rows; ++row) {
        for (int j = 0; j < num_cols_B; ++j) {
            float sum = 0.0f;
            for (int idx = h_rowPtr[row]; idx < h_rowPtr[row + 1]; ++idx) {
                sum += h_val[idx] * h_B[h_colIdx[idx] * num_cols_B + j];
            }
            h_C[row * num_cols_B + j] = sum;
        }
    }
}

bool compare_results(const std::vector<float> &gpu_result, const std::vector<float> &cpu_result) {
    if (gpu_result.size() != cpu_result.size()) return false;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

