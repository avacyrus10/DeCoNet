#include "ifft.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

void create_directory(const std::string& dir) {
    if (access(dir.c_str(), F_OK) == -1) {
        if (mkdir(dir.c_str(), 0777) != 0) {
            std::cerr << "Failed to create directory: " << dir << std::endl;
            exit(1);
        }
    }
}

void create_constant_input(const std::string& filename, int N) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        outfile << "1 0\n";
    }
}

void create_impulse_input(const std::string& filename, int N) {
    std::ofstream outfile(filename);
    outfile << "1 0\n";
    for (int i = 1; i < N; i++) {
        outfile << "0 0\n";
    }
}

void create_sine_wave_input(const std::string& filename, int N, float frequency, float sampling_rate) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        outfile << sin(2.0f * M_PI * frequency * i / sampling_rate) << " 0\n";
    }
}

void create_cosine_wave_input(const std::string& filename, int N, float frequency, float sampling_rate) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        outfile << cos(2.0f * M_PI * frequency * i / sampling_rate) << " 0\n";
    }
}

void create_sparse_input(const std::string& filename, int N, float sparsity, const std::vector<float>& values) {
    std::ofstream outfile(filename);
    srand(0);
    for (int i = 0; i < N; i++) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            int idx = rand() % values.size();
            outfile << values[idx] << " 0\n";
        } else {
            outfile << "0 0\n";
        }
    }
}

void read_input_from_file(const std::string& filename, float2 *data, int N) {
    std::ifstream infile(filename);
    for (int i = 0; i < N; i++) {
        infile >> data[i].x >> data[i].y;
    }
}

void validate_fft_ifft(const std::string &test_name, float2 *data, int N, bool use_cufft) {
    float2 *original_data;
    cudaMallocManaged(&original_data, N * sizeof(float2));
    cudaMemcpy(original_data, data, N * sizeof(float2), cudaMemcpyDefault);

    if (use_cufft) {
        cufft_fft(data, N);
        cufft_ifft(data, N);
    } else {
        cufft_fft(data, N);
        custom_ifft(data, N);
    }

    for (int i = 0; i < N; i++) {
        if (fabs(original_data[i].x - data[i].x) > 1e-5 || fabs(original_data[i].y - data[i].y) > 1e-5) {
            cudaFree(original_data);
            std::cout << test_name << " test failed!\n";
            return;
        }
    }

    cudaFree(original_data);
    std::cout << test_name << " test passed!\n";
}

void generate_inputs(const std::string &data_dir, int N) {
    create_directory(data_dir);

    create_constant_input(data_dir + "/constant_input.txt", N);
    create_impulse_input(data_dir + "/impulse_input.txt", N);
    create_sine_wave_input(data_dir + "/sine_wave_input.txt", N, 5.0f, 1024.0f);
    create_cosine_wave_input(data_dir + "/cosine_wave_input.txt", N, 5.0f, 1024.0f);

    std::vector<float> values = {1.0f, -1.0f};
    create_sparse_input(data_dir + "/sparse_input_0.1.txt", N, 0.1f, values);
    create_sparse_input(data_dir + "/sparse_input_0.2.txt", N, 0.2f, values);
    create_sparse_input(data_dir + "/sparse_input_0.3.txt", N, 0.3f, values);
}

int main(int argc, char *argv[]) {
    const int N = 1024;
    const std::string data_dir = "data";

    if (argc == 2 && std::string(argv[1]) == "generate") {
        generate_inputs(data_dir, N);
        std::cout << "Input files generated in " << data_dir << std::endl;
        return 0;
    }

    float2 *data;
    cudaMallocManaged(&data, N * sizeof(float2));

    for (const auto &filename : {"constant_input.txt", "impulse_input.txt", "sine_wave_input.txt", "cosine_wave_input.txt"}) {
        read_input_from_file(data_dir + "/" + filename, data, N);
        validate_fft_ifft(std::string(filename) + " FFT/IFFT", data, N, false);
        validate_fft_ifft(std::string(filename) + " cuFFT/IFFT", data, N, true);
    }

    for (float sparsity : {0.1f, 0.2f, 0.3f}) {
        std::string filename = "sparse_input_" + std::to_string(sparsity) + ".txt";
        read_input_from_file(data_dir + "/" + filename, data, N);
        validate_fft_ifft(filename + " FFT/IFFT", data, N, false);
        validate_fft_ifft(filename + " cuFFT/IFFT", data, N, true);
    }

    cudaFree(data);
    return 0;
}

