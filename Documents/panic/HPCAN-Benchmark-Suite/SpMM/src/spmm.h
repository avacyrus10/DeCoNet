#ifndef SPMM_H
#define SPMM_H

#include <vector>

void spmm(const std::vector<float> &d_val, const std::vector<int> &d_rowPtr, const std::vector<int> &d_colIdx,
          const std::vector<float> &d_B, std::vector<float> &d_C, int num_rows, int num_cols, int num_cols_B);

void spmm_cpu(const std::vector<float> &h_val, const std::vector<int> &h_rowPtr, const std::vector<int> &h_colIdx,
              const std::vector<float> &h_B, std::vector<float> &h_C, int num_rows, int num_cols_B);

bool compare_results(const std::vector<float> &gpu_result, const std::vector<float> &cpu_result);

#endif 

