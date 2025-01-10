#include "fft.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

int main() {
    const int N = 1024;
    float2 *data;
    
    cudaMallocManaged(&data, N * sizeof(float2));

    std::ifstream infile("data");
    if (!infile) {
        std::cerr << "Failed to open data file" << std::endl;
        return 1;
    }

    for (int i = 0; i < N; i++) {
        infile >> data[i].x >> data[i].y;
    }
    infile.close();

    fft(data, N);  
    ifft(data, N); 

    for (int i = 0; i < 10; i++) {
        std::cout << "Data[" << i << "] = " << data[i].x << " + " << data[i].y << "i" << std::endl;
    }

    cudaFree(data);
    return 0;
}

