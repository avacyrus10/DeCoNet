#include <iostream>
#include <vector>
#include "utils.h"

#ifdef USE_CUDNN
void cudnnConvolve3D(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize);
#else
void convolve3D(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize);
#endif

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <kernel_file> <output_file> <width> <height> <depth> <kernel_size> <use_cudnn>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string kernelFile = argv[2];
    std::string outputFile = argv[3];
    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);
    int depth = std::stoi(argv[6]);
    int kernelSize = std::stoi(argv[7]);
    bool useCudnn = std::stoi(argv[8]);

    int inputSize = width * height * depth;
    int kernelSize3D = kernelSize * kernelSize * kernelSize;
    int outputWidth = width - kernelSize + 1;
    int outputHeight = height - kernelSize + 1;
    int outputDepth = depth - kernelSize + 1;
    int outputSize = outputWidth * outputHeight * outputDepth;

    std::vector<float> input = read3DArrayFromFile(inputFile, inputSize);
    std::vector<float> kernel = read3DArrayFromFile(kernelFile, kernelSize3D);
    std::vector<float> output(outputSize);

#ifdef USE_CUDNN
    cudnnConvolve3D(input.data(), kernel.data(), output.data(), width, height, depth, kernelSize);
#else
    convolve3D(input.data(), kernel.data(), output.data(), width, height, depth, kernelSize);
#endif

    write3DArrayToFile(outputFile, output.data(), outputSize);

    return 0;
}

