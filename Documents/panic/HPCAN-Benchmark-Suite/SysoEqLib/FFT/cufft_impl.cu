#include "fft.h"
#include <cufft.h>

void cufft_fft(float2 *data, int N) {
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(data), reinterpret_cast<cufftComplex*>(data), CUFFT_FORWARD);
    cufftDestroy(plan);
}

void cufft_ifft(float2 *data, int N) {
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(data), reinterpret_cast<cufftComplex*>(data), CUFFT_INVERSE);
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        data[i].x /= N;
        data[i].y /= N;
    }
    cufftDestroy(plan);
}

