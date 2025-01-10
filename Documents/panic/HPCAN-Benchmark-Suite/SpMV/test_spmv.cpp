#include "spmv.h"
#include <iostream>
#include <vector>

void test_spmv() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<int> extras = {1, 2, 4, 8, 16};
    std::vector<float> sparsities = {0.9f, 0.95f, 0.99f};
    float tolerance = 1e-4;

    for (int size : sizes) {
        for (int extra : extras) {
            int rows = size + extra;
            int cols = size + extra;

            for (float sparsity : sparsities) {
                std::string folder = "data/" + std::to_string(rows) + "x" + std::to_string(cols) + "_sparsity_" + std::to_string(static_cast<int>(sparsity * 100));

                generate_sparse_matrix(rows, cols, sparsity, folder);
                generate_dense_vector(cols, folder);

                std::vector<float> val;
                std::vector<int> rowPtr, colIdx;
                read_sparse_matrix(folder, val, rowPtr, colIdx);

                std::vector<float> x;
                read_dense_vector(folder, x, cols);

                std::vector<float> y(rows, 0.0f);
                std::vector<float> y_cpu(rows, 0.0f);

                spmv(val, rowPtr, colIdx, x, y, rows);
                spmv_cpu(val, rowPtr, colIdx, x, y_cpu, rows);

                if (validate(y, y_cpu, tolerance)) {
                    std::cout << "Test passed for size " << rows << "x" << cols << " with sparsity " << sparsity << "\n";
                } else {
                    std::cerr << "Test failed for size " << rows << "x" << cols << " with sparsity " << sparsity << "\n";
                }
            }
        }
    }
}

int main() {
    test_spmv();
    return 0;
}

