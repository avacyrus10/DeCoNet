#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "convolution2d.h"

#define CHECK_CUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void convolve2D_cudnn(const float* input, const float* kernel, float* output, int width, int height, int kernelSize) {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, height - kernelSize + 1, width - kernelSize + 1));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; 

    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize));

    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspaceSize);

    float *d_input, *d_kernel, *d_output;
    size_t inputSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);
    size_t outputSize = (width - kernelSize + 1) * (height - kernelSize + 1) * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_output));

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(kernelDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
}

