#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "spmm.h"

void read_sparse_matrix(const std::string &folder, std::vector<float> &val, std::vector<int> &rowPtr, std::vector<int> &colIdx) {
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

    val_file.close();
    rowPtr_file.close();
    colIdx_file.close();
}

void read_dense_matrix(const std::string &folder, std::vector<float> &B, int rows, int cols) {
    std::ifstream B_file(folder + "/B.bin", std::ios::binary);
    B.resize(rows * cols);
    B_file.read(reinterpret_cast<char*>(B.data()), B.size() * sizeof(float));
    B_file.close();
}

bool validate_results(const std::vector<float> &gpu_result, const std::vector<float> &cpu_result, float tolerance = 1e-4) {
    if (gpu_result.size() != cpu_result.size()) return false;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU = " << gpu_result[i] << ", CPU = " << cpu_result[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    std::vector<int> extras = {1, 2, 4, 8, 16};
    std::vector<float> sparsities = {0.9f, 0.95f, 0.99f};

    for (int size : sizes) {
        for (int extra : extras) {
            int rows = size + extra;
            int cols = size + extra;
            for (float sparsity : sparsities) {
                std::string folder = "data/" + std::to_string(rows) + "x" + std::to_string(cols) +
                                     "_sparsity_" + std::to_string(static_cast<int>(sparsity * 100));
                
                std::vector<float> val, B, C(rows * rows), C_cpu(rows * rows);
                std::vector<int> rowPtr, colIdx;

                read_sparse_matrix(folder, val, rowPtr, colIdx);
                read_dense_matrix(folder, B, cols, rows);

                spmm(val, rowPtr, colIdx, B, C, rows, cols, rows);
                spmm_cpu(val, rowPtr, colIdx, B, C_cpu, rows, rows);

                if (validate_results(C, C_cpu)) {
                    std::cout << "Validation passed for " << folder << "\n";
                } else {
                    std::cout << "Validation failed for " << folder << "\n";
                }
            }
        }
    }

    return 0;
}

