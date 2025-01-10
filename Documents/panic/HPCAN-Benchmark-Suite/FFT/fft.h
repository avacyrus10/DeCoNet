#ifndef FFT_H
#define FFT_H

#include <cuda_runtime.h>

// Function declarations for custom FFT
void fft(float2 *data, int N);
void ifft(float2 *data, int N);

// Function declarations for cuFFT
void cufft_fft(float2 *data, int N);
void cufft_ifft(float2 *data, int N);

#endif 

