#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" void dct(const float* d_input, float* d_output, int N);
extern "C" void idct(const float* d_input, float* d_output, int N);

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

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
        outfile << "1\n";
    }
    outfile.close();
}

void create_impulse_input(const std::string& filename, int N) {
    std::ofstream outfile(filename);
    outfile << "1\n";
    for (int i = 1; i < N; i++) {
        outfile << "0\n";
    }
    outfile.close();
}

void create_sine_wave_input(const std::string& filename, int N, float frequency, float sampling_rate) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        outfile << sin(2.0f * M_PI * frequency * i / sampling_rate) << "\n";
    }
    outfile.close();
}

void create_cosine_wave_input(const std::string& filename, int N, float frequency, float sampling_rate) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        outfile << cos(2.0f * M_PI * frequency * i / sampling_rate) << "\n";
    }
    outfile.close();
}

void create_random_input(const std::string& filename, int N) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        outfile << static_cast<float>(rand()) / RAND_MAX << "\n";
    }
    outfile.close();
}

void create_sparse_input(const std::string& filename, int N, float sparsity, const std::vector<float>& values) {
    std::ofstream outfile(filename);
    for (int i = 0; i < N; i++) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            outfile << values[rand() % values.size()] << "\n";
        } else {
            outfile << "0\n";
        }
    }
    outfile.close();
}

void generate_inputs(int N) {
    create_directory("data");

    // Ensure all inputs are created
    create_constant_input("data/constant_input.txt", N);
    create_impulse_input("data/impulse_input.txt", N);
    create_sine_wave_input("data/sine_wave_input.txt", N, 5.0f, 1024.0f);
    create_cosine_wave_input("data/cosine_wave_input.txt", N, 5.0f, 1024.0f);
    create_random_input("data/random_input.txt", N);

    // Sparse inputs
    std::vector<float> values = {1.0f, -1.0f};
    for (float sparsity : {0.1f, 0.2f, 0.3f}) {
        create_sparse_input("data/sparse_sine_wave_input_" + std::to_string(sparsity) + ".txt", N, sparsity, values);
        create_sparse_input("data/sparse_cosine_wave_input_" + std::to_string(sparsity) + ".txt", N, sparsity, values);
    }
}

void read_input_from_file(const std::string& filename, float* data, int N) {
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

void test_dct_idct(const std::string& description, const std::string& filename, int N, float tolerance = 1e-3) {
    float *h_input, *h_dct_output, *h_idct_output;
    float *d_input, *d_dct_output, *d_idct_output;

    h_input = (float*)malloc(N * sizeof(float));
    h_dct_output = (float*)malloc(N * sizeof(float));
    h_idct_output = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_dct_output, N * sizeof(float));
    cudaMalloc(&d_idct_output, N * sizeof(float));

    read_input_from_file(filename, h_input, N);
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dct(d_input, d_dct_output, N);
    idct(d_dct_output, d_idct_output, N);

    cudaMemcpy(h_idct_output, d_idct_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_input[i] - h_idct_output[i]) >= tolerance) {
            std::cerr << description << " Mismatch at index " << i << ": expected " << h_input[i]
                      << ", got " << h_idct_output[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << description << " DCT/IDCT test passed!" << std::endl;
    } else {
        std::cerr << description << " DCT/IDCT test failed!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_dct_output);
    cudaFree(d_idct_output);
    free(h_input);
    free(h_dct_output);
    free(h_idct_output);
}

int main(int argc, char* argv[]) {
    const int N = 1024;

    if (argc != 2) {
        std::cerr << "Usage: ./dct_idct_test --generate | --validate" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "--generate") {
        generate_inputs(N);
        std::cout << "Input files generated in the 'data/' directory." << std::endl;
    } else if (mode == "--validate") {
        test_dct_idct("Constant input", "data/constant_input.txt", N);
        test_dct_idct("Impulse input", "data/impulse_input.txt", N);
        test_dct_idct("Sine wave input", "data/sine_wave_input.txt", N);
        test_dct_idct("Cosine wave input", "data/cosine_wave_input.txt", N);
        test_dct_idct("Random input", "data/random_input.txt", N);

        for (float sparsity : {0.1f, 0.2f, 0.3f}) {
            test_dct_idct("Sparse sine wave input (sparsity " + std::to_string(sparsity) + ")",
                          "data/sparse_sine_wave_input_" + std::to_string(sparsity) + ".txt", N);
            test_dct_idct("Sparse cosine wave input (sparsity " + std::to_string(sparsity) + ")",
                          "data/sparse_cosine_wave_input_" + std::to_string(sparsity) + ".txt", N);
        }

        std::cout << "Validation completed successfully." << std::endl;
    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        std::cerr << "Usage: ./dct_idct_test --generate | --validate" << std::endl;
        return 1;
    }

    return 0;
}

