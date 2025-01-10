#include "ifft.h"
#include <cufft.h>
#include <iostream>

void cufft_fft(float2 *data, int N) {
    cufftHandle plan;
    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        std::cerr << "Error: Failed to create cuFFT plan for FFT" << std::endl;
        return;
    }

    if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(data), reinterpret_cast<cufftComplex*>(data), CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::cerr << "Error: Failed to execute cuFFT FFT" << std::endl;
        cufftDestroy(plan);
        return;
    }

    cudaDeviceSynchronize();
    cufftDestroy(plan);
}

void cufft_ifft(float2 *data, int N) {
    cufftHandle plan;
    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        std::cerr << "Error: Failed to create cuFFT plan for IFFT" << std::endl;
        return;
    }

    if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(data), reinterpret_cast<cufftComplex*>(data), CUFFT_INVERSE) != CUFFT_SUCCESS) {
        std::cerr << "Error: Failed to execute cuFFT IFFT" << std::endl;
        cufftDestroy(plan);
        return;
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        data[i].x /= N;
        data[i].y /= N;
    }

    cufftDestroy(plan);
}

