#ifndef CORRELATION3D_H
#define CORRELATION3D_H

#include <vector>
#include <string>

void correlate3D(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize);
std::vector<float> read3DArrayFromFile(const std::string& filename, int size);
void write3DArrayToFile(const std::string& filename, const float* array, int size);

#endif

