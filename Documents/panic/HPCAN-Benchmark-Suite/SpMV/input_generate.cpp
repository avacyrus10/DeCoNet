#include "spmv.h"
#include <iostream>
#include <vector>

void generate_inputs() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<int> extras = {1, 2, 4, 8, 16};
    std::vector<float> sparsities = {0.9f, 0.95f, 0.99f};

    for (int size : sizes) {
        for (int extra : extras) {
            int rows = size + extra;
            int cols = size + extra;

            for (float sparsity : sparsities) {

                std::string folder = "data/" + std::to_string(rows) + "x" + std::to_string(cols) + "_sparsity_" + std::to_string(static_cast<int>(sparsity * 100));

                generate_sparse_matrix(rows, cols, sparsity, folder);
                generate_dense_vector(cols, folder);

                std::cout << "Generated inputs for size " << rows << "x" << cols << " with sparsity " << sparsity << " in folder: " << folder << "\n";
            }
        }
    }
}

int main() {
    generate_inputs();
    std::cout << "Input generation complete.\n";
    return 0;
}

