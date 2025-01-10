#include "dct.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cassert>

void create_sine_wave_input(const std::string& filename, int N, float frequency, float sampling_rate) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        outfile << sin(2.0f * M_PI * frequency * i / sampling_rate) << "\n";
    }
    outfile.close();
}

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

void test_dct_sine_wave_input() {
    const int N = 1024;
    float *data;
    float *d_data, *d_result, *result;

    data = (float*)malloc(N * sizeof(float));
    result = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    read_input_from_file("data/sine_wave_input.txt", data, N);
    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

    dct(d_data, d_result, N);
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << "DCT Result[" << i << "] = " << result[i] << std::endl;
    }

    idct(d_result, d_data, N);
    cudaMemcpy(result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        assert(fabs(result[i] - data[i]) < 1e-5);
    }

    cudaFree(d_data);
    cudaFree(d_result);
    free(data);
    free(result);
    std::cout << "Sine wave input DCT/IDCT test passed!" << std::endl;
}

int main() {
    const int N = 1024;
    create_sine_wave_input("data/sine_wave_input.txt", N, 5.0f, 1024.0f);

    test_dct_sine_wave_input();

    return 0;
}

