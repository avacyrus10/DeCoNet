#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

std::vector<float> read3DArrayFromFile(const std::string& filename, int size);
void write3DArrayToFile(const std::string& filename, const float* array, int size);

#endif

