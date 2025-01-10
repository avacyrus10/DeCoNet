#include "sma.h"
#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>
#include <cmath>

extern "C" {

// Input generation
void generate_sparse_matrix(int rows, int cols, float sparsity, const std::string &folder) {
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

    std::ofstream(folder + "/val.bin", std::ios::binary).write(reinterpret_cast<const char *>(val.data()), val.size() * sizeof(float));
    std::ofstream(folder + "/rowPtr.bin", std::ios::binary).write(reinterpret_cast<const char *>(rowPtr.data()), rowPtr.size() * sizeof(int));
    std::ofstream(folder + "/colIdx.bin", std::ios::binary).write(reinterpret_cast<const char *>(colIdx.data()), colIdx.size() * sizeof(int));
}

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

    val_file.read(reinterpret_cast<char *>(val.data()), val.size() * sizeof(float));
    rowPtr_file.read(reinterpret_cast<char *>(rowPtr.data()), rowPtr.size() * sizeof(int));
    colIdx_file.read(reinterpret_cast<char *>(colIdx.data()), colIdx.size() * sizeof(int));
}

// SMA CPU computation
void sma_cpu(const std::vector<float>& h_valA, const std::vector<int>& h_rowPtrA, const std::vector<int>& h_colIdxA,
             const std::vector<float>& h_valB, const std::vector<int>& h_rowPtrB, const std::vector<int>& h_colIdxB,
             std::vector<float>& h_valC, std::vector<int>& h_rowPtrC, std::vector<int>& h_colIdxC, int num_rows) {
    for (int row = 0; row < num_rows; ++row) {
        int idxC = h_rowPtrC[row];
        int idxA = h_rowPtrA[row];
        int idxB = h_rowPtrB[row];

        while (idxA < h_rowPtrA[row + 1] && idxB < h_rowPtrB[row + 1]) {
            int colA = h_colIdxA[idxA];
            int colB = h_colIdxB[idxB];

            if (colA == colB) {
                h_valC[idxC] = h_valA[idxA] + h_valB[idxB];
                h_colIdxC[idxC] = colA;
                idxA++;
                idxB++;
            } else if (colA < colB) {
                h_valC[idxC] = h_valA[idxA];
                h_colIdxC[idxC] = colA;
                idxA++;
            } else {
                h_valC[idxC] = h_valB[idxB];
                h_colIdxC[idxC] = colB;
                idxB++;
            }
            idxC++;
        }

        while (idxA < h_rowPtrA[row + 1]) {
            h_valC[idxC] = h_valA[idxA];
            h_colIdxC[idxC] = h_colIdxA[idxA];
            idxA++;
            idxC++;
        }

        while (idxB < h_rowPtrB[row + 1]) {
            h_valC[idxC] = h_valB[idxB];
            h_colIdxC[idxC] = h_colIdxB[idxB];
            idxB++;
            idxC++;
        }
    }
}

// Validation
bool validate(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, float tolerance) {
    if (gpu_result.size() != cpu_result.size()) return false;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": GPU result " << gpu_result[i]
                      << ", CPU result " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

} // extern "C"

