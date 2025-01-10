#include <cuda_runtime.h>
#include <cusparse.h>
#include <filesystem>
#include <random>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
{                                                                               \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                  << cudaGetErrorString(err) << std::endl;                      \
        exit(1);                                                                \
    }                                                                           \
}

extern "C" {

void spmm(const std::vector<float>& h_val, const std::vector<int>& h_rowPtr, 
          const std::vector<int>& h_colIdx, const std::vector<float>& h_B, 
          std::vector<float>& h_C, int num_rows, int num_cols, int num_cols_B) {

    float *d_val, *d_B, *d_C;
    int *d_rowPtr, *d_colIdx;

    CUDA_CHECK(cudaMalloc(&d_val, h_val.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, h_rowPtr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, h_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), h_val.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr.data(), h_rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    cusparseCreateCsr(&matA, num_rows, num_cols, h_val.size(), d_rowPtr, d_colIdx, d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnMat(&matB, num_cols, num_cols_B, num_cols_B, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, num_rows, num_cols_B, num_cols_B, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 0.0f;
    void *d_buffer = nullptr;
    size_t bufferSize = 0;

    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void spmm_cpu(const std::vector<float>& h_val, const std::vector<int>& h_rowPtr,
              const std::vector<int>& h_colIdx, const std::vector<float>& h_B,
              std::vector<float>& h_C, int num_rows, int num_cols_B) {
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

bool compare_results(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result) {
    if (gpu_result.size() != cpu_result.size()) return false;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > 1e-4) {
            std::cerr << "Mismatch at index " << i << ": GPU result " << gpu_result[i] 
                      << ", CPU result " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

void generate_sparse_matrix(int rows, int cols, float sparsity, const std::string& folder) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0, 1);

    std::vector<float> val;
    std::vector<int> rowPtr = {0};
    std::vector<int> colIdx;
    int nnz = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) > sparsity) {
                val.push_back(dis(gen));
                colIdx.push_back(j);
                nnz++;
            }
        }
        rowPtr.push_back(nnz);
    }

    std::filesystem::create_directories(folder);

    std::ofstream(folder + "/val.bin", std::ios::binary).write(reinterpret_cast<const char*>(val.data()), val.size() * sizeof(float));
    std::ofstream(folder + "/rowPtr.bin", std::ios::binary).write(reinterpret_cast<const char*>(rowPtr.data()), rowPtr.size() * sizeof(int));
    std::ofstream(folder + "/colIdx.bin", std::ios::binary).write(reinterpret_cast<const char*>(colIdx.data()), colIdx.size() * sizeof(int));
}

void generate_and_save_dense_matrix(int rows, int cols, const std::string& folder) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0, 1);

    std::vector<float> B(rows * cols);
    for (float& val : B) val = dis(gen);

    std::ofstream(folder + "/B.bin", std::ios::binary).write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(float));
}

void read_sparse_matrix(const std::string& folder, std::vector<float>& val, std::vector<int>& rowPtr, std::vector<int>& colIdx) {
    std::ifstream val_file(folder + "/val.bin", std::ios::binary);
    std::ifstream rowPtr_file(folder + "/rowPtr.bin", std::ios::binary);
    std::ifstream colIdx_file(folder + "/colIdx.bin", std::ios::binary);

    val_file.seekg(0, std::ios::end);
    rowPtr_file.seekg(0, std::ios::end);
    colIdx_file.seekg(0, std::ios::end);

    size_t val_size = val_file.tellg() / sizeof(float);
    size_t rowPtr_size = rowPtr_file.tellg() / sizeof(int);
    size_t colIdx_size = colIdx_file.tellg() / sizeof(int);

    val.resize(val_size);
    rowPtr.resize(rowPtr_size);
    colIdx.resize(colIdx_size);

    val_file.seekg(0, std::ios::beg);
    rowPtr_file.seekg(0, std::ios::beg);
    colIdx_file.seekg(0, std::ios::beg);

    val_file.read(reinterpret_cast<char*>(val.data()), val.size() * sizeof(float));
    rowPtr_file.read(reinterpret_cast<char*>(rowPtr.data()), rowPtr.size() * sizeof(int));
    colIdx_file.read(reinterpret_cast<char*>(colIdx.data()), colIdx.size() * sizeof(int));
}

void read_dense_matrix(const std::string& folder, std::vector<float>& B, int rows, int cols) {
    B.resize(rows * cols);
    std::ifstream(folder + "/B.bin", std::ios::binary).read(reinterpret_cast<char*>(B.data()), B.size() * sizeof(float));
}

}

