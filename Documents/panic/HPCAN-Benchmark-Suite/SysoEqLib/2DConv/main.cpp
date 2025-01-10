#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "convolution2d.h"

std::vector<float> read2DArrayFromFile(const std::string& filename, int size) {
    std::vector<float> array(size);
    std::ifstream file(filename);
    for (int i = 0; i < size; ++i) {
        file >> array[i];
    }
    file.close();
    return array;
}

void write2DArrayToFile(const std::string& filename, const float* array, int size) {
    std::ofstream file(filename);
    for (int i = 0; i < size; ++i) {
        file << array[i] << " ";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        return 1;
    }

    std::string inputFile = argv[1];
    std::string kernelFile = argv[2];
    std::string outputFile = argv[3];
    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);
    int kernelSize = std::stoi(argv[6]);

    int inputSize = width * height;
    int kernelSize2D = kernelSize * kernelSize;
    int outputWidth = width - kernelSize + 1;
    int outputHeight = height - kernelSize + 1;
    int outputSize = outputWidth * outputHeight;

    std::vector<float> input = read2DArrayFromFile(inputFile, inputSize);
    std::vector<float> kernel = read2DArrayFromFile(kernelFile, kernelSize2D);
    std::vector<float> output(outputSize);

#ifdef USE_CUDNN
    convolve2D_cudnn(input.data(), kernel.data(), output.data(), width, height, kernelSize);
#else
    convolve2D(input.data(), kernel.data(), output.data(), width, height, kernelSize);
#endif

    write2DArrayToFile(outputFile, output.data(), outputSize);

    return 0;
}

