#ifndef SPMV_H
#define SPMV_H

#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void spmv(const std::vector<float>& val, const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
          const std::vector<float>& x, std::vector<float>& y, int num_rows);

void spmv_cpu(const std::vector<float>& val, const std::vector<int>& rowPtr, const std::vector<int>& colIdx,
              const std::vector<float>& x, std::vector<float>& y, int num_rows);

bool validate(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, float tolerance);

void generate_sparse_matrix(int rows, int cols, float sparsity, const std::string& folder);

void generate_dense_vector(int size, const std::string& folder);

void read_sparse_matrix(const std::string& folder, std::vector<float>& val, std::vector<int>& rowPtr, std::vector<int>& colIdx);

void read_dense_vector(const std::string& folder, std::vector<float>& x, int size);

#ifdef __cplusplus
}
#endif

#endif

