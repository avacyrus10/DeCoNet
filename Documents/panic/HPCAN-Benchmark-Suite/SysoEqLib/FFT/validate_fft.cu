#include "fft.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>  
#include <iomanip>   
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

void test_fft_constant_input() {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input("data/constant_input.txt", data, N);
    fft(data, N);
    ifft(data, N);

    for (int i = 0; i < N; i++) {
        assert(fabs(data[i].x - 1.0f) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "Constant input FFT/IFFT test passed!" << std::endl;
}

void test_fft_impulse_input() {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input("data/impulse_input.txt", data, N);
    fft(data, N);
    ifft(data, N);

    for (int i = 0; i < N; i++) {
        assert(fabs(data[i].x - ((i == 0) ? 1.0f : 0.0f)) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << "Impulse input FFT/IFFT test passed!" << std::endl;
}

void test_fft_wave_input(const std::string &filename, bool is_sine, float frequency) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input(filename, data, N);
    fft(data, N);
    ifft(data, N);

    for (int i = 0; i < N; i++) {
        float expected = is_sine ? sin(2.0f * M_PI * frequency * i / N)
                                 : cos(2.0f * M_PI * frequency * i / N);
        assert(fabs(data[i].x - expected) < 1e-5);
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << (is_sine ? "Sine" : "Cosine") << " wave input FFT/IFFT test passed!" << std::endl;
}

void test_fft_with_noise(const std::string &filename, bool is_gaussian, float mean, float stddev) {
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

    fft(data, N);
    ifft(data, N);

    for (int i = 0; i < N; i++) {
        float expected = original_signal[i] + added_noise[i]; 
        if (fabs(data[i].x - expected) >= 1e-3) {             
            std::cerr << "Mismatch at index " << i << ": "
                      << "expected=" << expected << ", got=" << data[i].x
                      << ", noise=" << added_noise[i] << ", original=" << original_signal[i] << std::endl;
            assert(false); 
        }
        assert(fabs(data[i].y) < 1e-3);  
    }

    cudaFree(data);
    std::cout << (is_gaussian ? "Gaussian" : "Uniform") << " noise FFT/IFFT test passed!" << std::endl;
}



void test_fft_sum_of_waves(const std::string &filename, bool is_sine, const std::vector<float> &frequencies) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    read_input(filename, data, N);
    fft(data, N);
    ifft(data, N);

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
    std::cout << (is_sine ? "Sum of sine waves" : "Sum of cosine waves") << " input FFT/IFFT test passed!" << std::endl;
}
void read_input_from_file(const std::string& filename, float2 *data, int N) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        infile >> data[i].x >> data[i].y;
    }
    infile.close();
}

void test_fft_sparse_input(const std::string& filename, float sparsity, const std::vector<float>& original_values) {
    const int N = 1024;
    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));
    read_input_from_file(filename, data, N);

    fft(data, N);
    ifft(data, N);

    srand(SEED); 
    for (int i = 0; i < N; i++) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            int idx = rand() % original_values.size();
            assert(fabs(data[i].x - original_values[idx]) < 1e-5);
        } else {
            assert(fabs(data[i].x) < 1e-5);
        }
        assert(fabs(data[i].y) < 1e-5);
    }

    cudaFree(data);
    std::cout << filename << " with sparsity " << sparsity << " FFT/IFFT test passed!" << std::endl;
}


void run_all_tests() {
    test_fft_constant_input();
    test_fft_impulse_input();
    test_fft_wave_input("data/sine_wave_input.txt", true, 5.0f);
    test_fft_wave_input("data/cosine_wave_input.txt", false, 5.0f);
    test_fft_with_noise("data/sine_wave_uniform_noise_input.txt", false, 0.0f, 0.0f);
    test_fft_with_noise("data/cosine_wave_gaussian_noise_input.txt", true, 0.0f, 0.1f);

    std::vector<float> frequencies = {5.0f, 10.0f, 15.0f};
    test_fft_sum_of_waves("data/sum_of_sine_waves_input.txt", true, frequencies);
    test_fft_sum_of_waves("data/sum_of_cosine_waves_input.txt", false, frequencies);

    std::vector<float> values = {1.0f, -1.0f};
    test_fft_sparse_input("data/sparse_sine_wave_input_0.100000.txt0.1.txt", 0.1f, values);
    test_fft_sparse_input("data/sparse_cosine_wave_input_0.200000.txt0.2.txt", 0.2f, values);
    test_fft_sparse_input("data/sparse_sum_of_sine_waves_input_0.300000.txt0.3.txt", 0.3f, frequencies);
    test_fft_sparse_input("data/sparse_sum_of_cosine_waves_input_0.300000.txt0.3.txt", 0.3f, frequencies);
}


int main() {
    run_all_tests();
    return 0;
}

