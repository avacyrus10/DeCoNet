#include "fft.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const unsigned int SEED = 0;

void read_input(const std::string &filename, float2 *data, int N) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        infile >> data[i].x >> data[i].y;
    }
    infile.close();
}

float generate_gaussian_noise(float mean, float stddev) {
    static bool has_spare = false;
    static float spare;

    if (has_spare) {
        has_spare = false;
        return mean + stddev * spare;
    }

    float u, v, s;
    do {
        u = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = true;

    return mean + stddev * u * s;
}

void test_cufft_constant_input() {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input("data/constant_input.txt", data, N);
    cufft_fft(data, N);
    cufft_ifft(data, N);

    for (int i = 0; i < N; i++) {
        assert(fabs(data[i].x - 1.0f) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "cuFFT - Constant input test passed!" << std::endl;
}

void test_cufft_impulse_input() {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input("data/impulse_input.txt", data, N);
    cufft_fft(data, N);
    cufft_ifft(data, N);

    for (int i = 0; i < N; i++) {
        assert(fabs(data[i].x - ((i == 0) ? 1.0f : 0.0f)) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "cuFFT - Impulse input test passed!" << std::endl;
}

void test_cufft_wave_input(const std::string &filename, bool is_sine, float frequency) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input(filename, data, N);
    cufft_fft(data, N);
    cufft_ifft(data, N);

    for (int i = 0; i < N; i++) {
        float expected = is_sine ? sin(2.0f * M_PI * frequency * i / N)
                                 : cos(2.0f * M_PI * frequency * i / N);
        assert(fabs(data[i].x - expected) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "cuFFT - " << (is_sine ? "Sine" : "Cosine") << " wave input test passed!" << std::endl;
}

void test_cufft_with_noise(const std::string &filename, bool is_gaussian, float mean, float stddev) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input(filename, data, N);

    std::vector<float> original_signal(N);
    std::vector<float> added_noise(N);
    srand(SEED);
    for (int i = 0; i < N; i++) {
        float noise = is_gaussian ? generate_gaussian_noise(mean, stddev)
                                  : 0.1f * ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 2.0f;
        original_signal[i] = data[i].x;
        added_noise[i] = noise;
        data[i].x += noise;
    }

    cufft_fft(data, N);
    cufft_ifft(data, N);

    for (int i = 0; i < N; i++) {
        float expected = original_signal[i] + added_noise[i];
        assert(fabs(data[i].x - expected) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "cuFFT - " << (is_gaussian ? "Gaussian" : "Uniform") << " noise test passed!" << std::endl;
}

void test_cufft_sum_of_waves(const std::string &filename, bool is_sine, const std::vector<float> &frequencies) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input(filename, data, N);
    cufft_fft(data, N);
    cufft_ifft(data, N);

    for (int i = 0; i < N; i++) {
        float expected = 0.0f;
        for (float frequency : frequencies) {
            expected += is_sine ? sin(2.0f * M_PI * frequency * i / N)
                                : cos(2.0f * M_PI * frequency * i / N);
        }
        assert(fabs(data[i].x - expected) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "cuFFT - " << (is_sine ? "Sum of sine waves" : "Sum of cosine waves") << " test passed!" << std::endl;
}

void test_cufft_sparse_input(const std::string &filename, float sparsity, const std::vector<float> &values) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input(filename, data, N);
    cufft_fft(data, N);
    cufft_ifft(data, N);

    srand(SEED);
    for (int i = 0; i < N; i++) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            int idx = rand() % values.size();
            assert(fabs(data[i].x - values[idx]) < 1e-5);
        } else {
            assert(fabs(data[i].x) < 1e-5);
        }
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "cuFFT - Sparse input " << filename << " test passed!" << std::endl;
}

void run_all_cufft_tests() {
    test_cufft_constant_input();
    test_cufft_impulse_input();
    test_cufft_wave_input("data/sine_wave_input.txt", true, 5.0f);
    test_cufft_wave_input("data/cosine_wave_input.txt", false, 5.0f);
    test_cufft_with_noise("data/sine_wave_uniform_noise_input.txt", false, 0.0f, 0.0f);
    test_cufft_with_noise("data/cosine_wave_gaussian_noise_input.txt", true, 0.0f, 0.1f);

    std::vector<float> frequencies = {5.0f, 10.0f, 15.0f};
    test_cufft_sum_of_waves("data/sum_of_sine_waves_input.txt", true, frequencies);
    test_cufft_sum_of_waves("data/sum_of_cosine_waves_input.txt", false, frequencies);

    std::vector<float> values = {1.0f, -1.0f};
    test_cufft_sparse_input("data/sparse_sine_wave_input_0.100000.txt0.1.txt", 0.1f, values);
    test_cufft_sparse_input("data/sparse_cosine_wave_input_0.200000.txt0.2.txt", 0.2f, values);
    test_cufft_sparse_input("data/sparse_sum_of_sine_waves_input_0.300000.txt0.3.txt", 0.3f, frequencies);
    test_cufft_sparse_input("data/sparse_sum_of_cosine_waves_input_0.300000.txt0.3.txt", 0.3f, frequencies);
}

int main() {
    run_all_cufft_tests();
    return 0;
    }
