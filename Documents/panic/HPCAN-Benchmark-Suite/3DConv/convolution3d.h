#ifndef CONVOLUTION3D_H
#define CONVOLUTION3D_H

void convolve3D(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize);

#endif

