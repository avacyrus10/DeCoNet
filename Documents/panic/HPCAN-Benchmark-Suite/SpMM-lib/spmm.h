#ifndef SPMM_H
#define SPMM_H

#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void spmm(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr, const std::vector<int> &h_colIdx,
          const std::vector<float> &h_B, std::vector<float> &h_C, int num_rows, int num_cols, int num_cols_B);

void spmm_cpu(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr, const std::vector<int> &h_colIdx,
              const std::vector<float> &h_B, std::vector<float> &h_C, int num_rows, int num_cols_B);

bool compare_results(const std::vector<float> &gpu_result, const std::vector<float> &cpu_result);

void generate_sparse_matrix(int rows, int cols, float sparsity, const std::string &folder);

void generate_and_save_dense_matrix(int rows, int cols, const std::string &folder);

void read_sparse_matrix(const std::string &folder, std::vector<float> &val, std::vector<int> &rowPtr, std::vector<int> &colIdx);

void read_dense_matrix(const std::string &folder, std::vector<float> &B, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif

