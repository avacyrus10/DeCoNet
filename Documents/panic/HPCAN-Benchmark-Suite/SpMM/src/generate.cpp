#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

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

    fs::create_directories(folder);

    std::ofstream val_file(folder + "/val.bin", std::ios::binary);
    std::ofstream rowPtr_file(folder + "/rowPtr.bin", std::ios::binary);
    std::ofstream colIdx_file(folder + "/colIdx.bin", std::ios::binary);

    val_file.write(reinterpret_cast<const char*>(val.data()), val.size() * sizeof(float));
    rowPtr_file.write(reinterpret_cast<const char*>(rowPtr.data()), rowPtr.size() * sizeof(int));
    colIdx_file.write(reinterpret_cast<const char*>(colIdx.data()), colIdx.size() * sizeof(int));

    val_file.close();
    rowPtr_file.close();
    colIdx_file.close();
}

void generate_and_save_dense_matrix(int rows, int cols, const std::string &folder) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dis(0, 1);

    std::vector<float> B(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        B[i] = dis(gen);
    }

    std::ofstream B_file(folder + "/B.bin", std::ios::binary);
    B_file.write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(float));
    B_file.close();
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
                generate_sparse_matrix(rows, cols, sparsity, folder);
                generate_and_save_dense_matrix(cols, rows, folder);
                std::cout << "Generated data for rows: " << rows << ", cols: " << cols
                          << ", sparsity: " << sparsity << ", folder: " << folder << "\n";
            }
        }
    }

    return 0;
}

