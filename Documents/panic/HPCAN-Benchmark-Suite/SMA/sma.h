#ifndef SMA_H
#define SMA_H

#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// Input generation
void generate_sparse_matrix(int rows, int cols, float sparsity, const std::string &folder);
void read_sparse_matrix(const std::string &folder, std::vector<float> &val, std::vector<int> &rowPtr, std::vector<int> &colIdx);

// SMA operations
void sma(const std::vector<float>& d_valA, const std::vector<int>& d_rowPtrA, const std::vector<int>& d_colIdxA,
         const std::vector<float>& d_valB, const std::vector<int>& d_rowPtrB, const std::vector<int>& d_colIdxB,
         std::vector<float>& d_valC, std::vector<int>& d_rowPtrC, std::vector<int>& d_colIdxC, int num_rows);

void sma_cpu(const std::vector<float>& h_valA, const std::vector<int>& h_rowPtrA, const std::vector<int>& h_colIdxA,
             const std::vector<float>& h_valB, const std::vector<int>& h_rowPtrB, const std::vector<int>& h_colIdxB,
             std::vector<float>& h_valC, std::vector<int>& h_rowPtrC, std::vector<int>& h_colIdxC, int num_rows);

// Validation
bool validate(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, float tolerance);

#ifdef __cplusplus
}
#endif

#endif
