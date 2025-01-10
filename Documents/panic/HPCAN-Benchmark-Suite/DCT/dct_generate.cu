#include "dct.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

const unsigned int SEED = 0;

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
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        outfile << "1\n";
    }
    outfile.close();
}

void create_impulse_input(const std::string& filename, int N) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    outfile << "1\n";
    for (int i = 1; i < N; i++) {
        outfile << "0\n";
    }
    outfile.close();
}

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

void create_cosine_wave_input(const std::string& filename, int N, float frequency, float sampling_rate) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        outfile << cos(2.0f * M_PI * frequency * i / sampling_rate) << "\n";
    }
    outfile.close();
}

void create_random_input(const std::string& filename, int N) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    srand(SEED);
    for (int i = 0; i < N; i++) {
        outfile << static_cast<float>(rand()) / RAND_MAX << "\n";
    }
    outfile.close();
}

void create_sine_wave_with_uniform_noise_input(const std::string& filename, int N, float frequency, float sampling_rate, float noise_amplitude) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    srand(SEED);
    for (int i = 0; i < N; i++) {
        float noise = noise_amplitude * ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 2.0f;
        outfile << sin(2.0f * M_PI * frequency * i / sampling_rate) + noise << "\n";
    }
    outfile.close();
}

void create_cosine_wave_with_uniform_noise_input(const std::string& filename, int N, float frequency, float sampling_rate, float noise_amplitude) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    srand(SEED);
    for (int i = 0; i < N; i++) {
        float noise = noise_amplitude * ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 2.0f;
        outfile << cos(2.0f * M_PI * frequency * i / sampling_rate) + noise << "\n";
    }
    outfile.close();
}

float generate_gaussian_noise(float mean, float stddev) {
    static bool has_spare = false;
    static float spare;
    if (has_spare) {
        has_spare = false;
        return mean + stddev * spare;
    }
    has_spare = true;
    static float u, v, s;
    do {
        u = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
}

void create_sine_wave_with_gaussian_noise_input(const std::string& filename, int N, float frequency, float sampling_rate, float noise_mean, float noise_stddev) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    srand(SEED);
    for (int i = 0; i < N; i++) {
        float noise = generate_gaussian_noise(noise_mean, noise_stddev);
        outfile << sin(2.0f * M_PI * frequency * i / sampling_rate) + noise << "\n";
    }
    outfile.close();
}

void create_cosine_wave_with_gaussian_noise_input(const std::string& filename, int N, float frequency, float sampling_rate, float noise_mean, float noise_stddev) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    srand(SEED);
    for (int i = 0; i < N; i++) {
        float noise = generate_gaussian_noise(noise_mean, noise_stddev);
        outfile << cos(2.0f * M_PI * frequency * i / sampling_rate) + noise << "\n";
    }
    outfile.close();
}

void create_sparse_input(const std::string& filename, int N, float sparsity, const std::vector<float>& values) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        exit(1);
    }
    srand(SEED);
    for (int i = 0; i < N; i++) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            int idx = rand() % values.size();
            outfile << values[idx] << "\n";
        } else {
            outfile << "0\n";
        }
    }
    outfile.close();
}

void create_input_files() {
    const int N = 1024;
    create_directory("data");
    create_constant_input("data/constant_input.txt", N);
    create_impulse_input("data/impulse_input.txt", N);
    create_sine_wave_input("data/sine_wave_input.txt", N, 5.0f, 1024.0f);
    create_cosine_wave_input("data/cosine_wave_input.txt", N, 5.0f, 1024.0f);
    create_sine_wave_with_uniform_noise_input("data/sine_wave_uniform_noise_input.txt", N, 5.0f, 1024.0f, 0.1f);
    create_cosine_wave_with_uniform_noise_input("data/cosine_wave_uniform_noise_input.txt", N, 5.0f, 1024.0f, 0.1f);
    create_sine_wave_with_gaussian_noise_input("data/sine_wave_gaussian_noise_input.txt", N, 5.0f, 1024.0f, 0.0f, 0.1f);
    create_cosine_wave_with_gaussian_noise_input("data/cosine_wave_gaussian_noise_input.txt", N, 5.0f, 1024.0f, 0.0f, 0.1f);
    create_random_input("data/random_input.txt", N);

    std::vector<float> values = {1.0f, -1.0f};
    std::vector<float> sparsity_levels = {0.1f, 0.2f, 0.3f};
    for (float sparsity : sparsity_levels) {
        create_sparse_input("data/sparse_sine_wave_input_" + std::to_string(sparsity) + ".txt", N, sparsity, values);
        create_sparse_input("data/sparse_cosine_wave_input_" + std::to_string(sparsity) + ".txt", N, sparsity, values);
    }

    std::cout << "Input files generated in 'data/' directory." << std::endl;
}

int main() {
    create_input_files();
    return 0;
}

