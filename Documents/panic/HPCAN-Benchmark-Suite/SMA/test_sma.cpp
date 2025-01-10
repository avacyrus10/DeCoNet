#include "sma.h"
#include <iostream>
#include <vector>

void test_sma() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    std::vector<int> extras = {1, 2, 4, 8, 16};
    std::vector<float> sparsities = {0.9f, 0.95f, 0.99f};
    float tolerance = 1e-4;

    for (int size : sizes) {
        for (int extra : extras) {
            int rows = size + extra;
            int cols = size + extra;

            for (float sparsity : sparsities) {
                std::string folderA = "data/A_" + std::to_string(rows) + "x" + std::to_string(cols) +
                                      "_sparsity_" + std::to_string(static_cast<int>(sparsity * 100));
                std::string folderB = "data/B_" + std::to_string(rows) + "x" + std::to_string(cols) +
                                      "_sparsity_" + std::to_string(static_cast<int>(sparsity * 100));

                std::vector<float> valA, valB, valC, C_cpu;
                std::vector<int> rowPtrA, rowPtrB, rowPtrC;
                std::vector<int> colIdxA, colIdxB, colIdxC;

                read_sparse_matrix(folderA, valA, rowPtrA, colIdxA);
                read_sparse_matrix(folderB, valB, rowPtrB, colIdxB);

                int nnzC = valA.size() + valB.size();
                valC.resize(nnzC);
                C_cpu.resize(nnzC);
                rowPtrC = rowPtrA;
                colIdxC.resize(nnzC);

                sma(valA, rowPtrA, colIdxA, valB, rowPtrB, colIdxB, valC, rowPtrC, colIdxC, rows);
                sma_cpu(valA, rowPtrA, colIdxA, valB, rowPtrB, colIdxB, C_cpu, rowPtrC, colIdxC, rows);

                if (validate(valC, C_cpu, tolerance)) {
                    std::cout << "Test passed for size " << rows << "x" << cols << " with sparsity " << sparsity << "\n";
                } else {
                    std::cerr << "Test failed for size " << rows << "x" << cols << " with sparsity " << sparsity << "\n";
                }
            }
        }
    }
}

int main() {
    test_sma();
    return 0;
}

