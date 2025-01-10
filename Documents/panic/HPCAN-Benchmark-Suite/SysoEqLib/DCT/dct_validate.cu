#include "dct.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

void read_input_from_file(const std::string& filename, float *data, int N) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        infile >> data[i];
    }
    infile.close();
}

void test_dct(const std::string& filename, const std::string& description, float tolerance = 1e-4) {
    const int N = 1024;
    float *data;
    float *d_data, *d_result, *result;

    data = (float*)malloc(N * sizeof(float));
    result = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    read_input_from_file(filename, data, N);
    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

    dct(d_data, d_result, N);
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    idct(d_result, d_data, N);
    cudaMemcpy(result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (fabs(result[i] - data[i]) >= tolerance) {
            std::cerr << "Mismatch at index " << i << " (" << description << "): "
                      << "expected " << data[i] << ", got " << result[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << description << " DCT/IDCT test passed!" << std::endl;
    } else {
        std::cerr << description << " DCT/IDCT test failed!" << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_result);
    free(data);
    free(result);
}

int main() {
    test_dct("data/constant_input.txt", "Constant input");
    test_dct("data/impulse_input.txt", "Impulse input");
    test_dct("data/sine_wave_input.txt", "Sine wave input");
    test_dct("data/cosine_wave_input.txt", "Cosine wave input");
    test_dct("data/sine_wave_uniform_noise_input.txt", "Sine wave with uniform noise input");
    test_dct("data/cosine_wave_uniform_noise_input.txt", "Cosine wave with uniform noise input");
    test_dct("data/sine_wave_gaussian_noise_input.txt", "Sine wave with Gaussian noise input");
    test_dct("data/cosine_wave_gaussian_noise_input.txt", "Cosine wave with Gaussian noise input");
    test_dct("data/random_input.txt", "Random input");

    std::vector<float> sparsity_levels = {0.1f, 0.2f, 0.3f};
    for (float sparsity : sparsity_levels) {
        test_dct("data/sparse_sine_wave_input_" + std::to_string(sparsity) + ".txt", 
                 "Sparse sine wave input (sparsity " + std::to_string(sparsity) + ")", 1e-3);
        test_dct("data/sparse_cosine_wave_input_" + std::to_string(sparsity) + ".txt", 
                 "Sparse cosine wave input (sparsity " + std::to_string(sparsity) + ")", 1e-3);
    }

    return 0;
}

