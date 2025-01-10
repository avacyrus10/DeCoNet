#include "spmm.h"
#include <iostream>
#include <vector>
#include <string>

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

                std::vector<float> val;
                std::vector<int> rowPtr;
                std::vector<int> colIdx;
                read_sparse_matrix(folder, val, rowPtr, colIdx);

                std::vector<float> B;
                read_dense_matrix(folder, B, cols, rows);

                std::vector<float> C(rows * rows, 0.0f);
                std::vector<float> C_cpu(rows * rows, 0.0f);

                spmm(val, rowPtr, colIdx, B, C, rows, cols, rows);
                spmm_cpu(val, rowPtr, colIdx, B, C_cpu, rows, rows);

                if (compare_results(C, C_cpu)) {
                    std::cout << "Test passed for size " << rows << "x" << cols << " with sparsity " << sparsity << "\n";
                } else {
                    std::cout << "Test failed for size " << rows << "x" << cols << " with sparsity " << sparsity << "\n";
                }
            }
        }
    }
    return 0;
}

