#include <iostream>
#include <vector>
#include "correlation3d.h"

int main(int argc, char* argv[]) {
    if (argc != 8) {
        return 1;
    }

    std::string inputFile = argv[1];
    std::string kernelFile = argv[2];
    std::string outputFile = argv[3];
    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);
    int depth = std::stoi(argv[6]);
    int kernelSize = std::stoi(argv[7]);

    int inputSize = width * height * depth;
    int kernelSize3D = kernelSize * kernelSize * kernelSize;
    int outputWidth = width - kernelSize + 1;
    int outputHeight = height - kernelSize + 1;
    int outputDepth = depth - kernelSize + 1;
    int outputSize = outputWidth * outputHeight * outputDepth;

    std::vector<float> input = read3DArrayFromFile(inputFile, inputSize);
    std::vector<float> kernel = read3DArrayFromFile(kernelFile, kernelSize3D);
    std::vector<float> output(outputSize);

    correlate3D(input.data(), kernel.data(), output.data(), width, height, depth, kernelSize);

    write3DArrayToFile(outputFile, output.data(), outputSize);

    return 0;
}
