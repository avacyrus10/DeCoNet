#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include <cmath>
#include <iostream>
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

void spmv(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr, const std::vector<int> &h_colIdx,
          const std::vector<float> &h_x, std::vector<float> &h_y, int num_rows) {

    float *d_val, *d_x, *d_y;
    int *d_rowPtr, *d_colIdx;

    CUDA_CHECK(cudaMalloc(&d_val, h_val.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, h_rowPtr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, h_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, h_y.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), h_val.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr.data(), h_rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    float alpha = 1.0f;
    float beta = 0.0f;

    #if CUSPARSE_VERSION >= 11000
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;

        cusparseCreateCsr(&matA, num_rows, h_x.size(), h_val.size(),
                          d_rowPtr, d_colIdx, d_val,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vecX, h_x.size(), d_x, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_rows, d_y, CUDA_R_32F);

        size_t bufferSize = 0;
        void *dBuffer = nullptr;

        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        CUDA_CHECK(cudaFree(dBuffer));

    #else

        cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, h_x.size(), h_val.size(),
                       &alpha, descrA, d_val, d_rowPtr, d_colIdx, d_x, &beta, d_y);
    #endif


    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(handle);
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

void spmv_cpu(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr, const std::vector<int> &h_colIdx,
              const std::vector<float> &h_x, std::vector<float> &h_y, int num_rows) {
    for (int row = 0; row < num_rows; ++row) {
        float sum = 0.0f;
        for (int idx = h_rowPtr[row]; idx < h_rowPtr[row + 1]; ++idx) {
            sum += h_val[idx] * h_x[h_colIdx[idx]];
        }
        h_y[row] = sum;
    }
}

bool compare_results(const std::vector<float> &gpu_result, const std::vector<float> &cpu_result) {
    if (gpu_result.size() != cpu_result.size()) return false;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > 1e-4) {
            std::cout << "Mismatch at index " << i << ": GPU result " << gpu_result[i]
                      << ", CPU result " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

