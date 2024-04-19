#include <iostream>
#include <vector>
#include <thrust/complex.h>
#include "fft_cuda.cuh" 

int main() {
    int size = 8;
    std::vector<thrust::complex<double>> input(size);
    for (int i = 0; i < size / 2; i++) {
        double realPart = rand() % 100 + 1;
        double imagPart = rand() % 100 + 1;
        input[i] = thrust::complex<double>(realPart, imagPart);
        input[size - 1 - i] = conj(input[i]); 
    }

    fft_cuda(input, false);

    std::cout << "FFT implementation:" << std::endl;
    for (const auto& value : input) {
        std::cout << value << std::endl;
    }

    return 0;
}
