#include "spmm.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    std::vector<int> extras = {1, 2, 4, 8};
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
                std::cout << "Generated input data in folder: " << folder << "\n";
            }
        }
    }
    return 0;
}

