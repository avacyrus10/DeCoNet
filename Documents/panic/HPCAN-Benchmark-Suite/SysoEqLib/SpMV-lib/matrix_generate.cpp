#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>
#include "spmv.h"

extern "C" {

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

void generate_dense_vector(int size, const std::string &folder) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0, 1);

    std::vector<float> x(size);
    for (float &val : x) val = dis(gen);

    std::ofstream(folder + "/x.bin", std::ios::binary).write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(float));
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

void read_dense_vector(const std::string &folder, std::vector<float> &x, int size) {
    x.resize(size);
    std::ifstream(folder + "/x.bin", std::ios::binary).read(reinterpret_cast<char *>(x.data()), x.size() * sizeof(float));
}

}

