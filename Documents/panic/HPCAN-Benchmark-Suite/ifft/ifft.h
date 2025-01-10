#ifndef IFFT_H
#define IFFT_H

#include <cuda_runtime.h>

void custom_ifft(float2 *data, int N);
void cufft_ifft(float2 *data, int N);
void cufft_fft(float2 *data, int N);

#endif

