#include <iostream>
#include <fstream>
#include <vector>
#include "correlation3d.h"

std::vector<float> read3DArrayFromFile(const std::string& filename, int size) {
    std::vector<float> array(size);
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return array;
    }
    for (int i = 0; i < size; ++i) {
        file >> array[i];
    }
    file.close();
    return array;
}

void write3DArrayToFile(const std::string& filename, const float* array, int size) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < size; ++i) {
        file << array[i] << " ";
    }
    file.close();
}

