#ifndef CORRELATION2D_H
#define CORRELATION2D_H

#include <vector>
#include <string>

void correlate2D(const float* input, const float* kernel, float* output, int width, int height, int kernelSize);
std::vector<float> read2DArrayFromFile(const std::string& filename, int size);
void write2DArrayToFile(const std::string& filename, const float* array, int size);

#endif
