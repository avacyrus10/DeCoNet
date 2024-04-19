#pragma once

#include <thrust/complex.h>
#include <vector>

void fft_cuda(std::vector<thrust::complex<double>>& a, bool invert);
