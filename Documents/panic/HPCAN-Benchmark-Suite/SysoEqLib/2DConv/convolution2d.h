#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

void convolve2D(const float* input, const float* kernel, float* output, int width, int height, int kernelSize);
void convolve2D_cudnn(const float* input, const float* kernel, float* output, int width, int height, int kernelSize);

#endif 

