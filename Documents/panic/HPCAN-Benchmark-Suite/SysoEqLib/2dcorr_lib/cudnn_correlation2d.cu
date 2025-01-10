#include <cudnn.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "correlation2d.h"

void cudnnCorrelate2D(const float* input, const float* kernel, float* output, int width, int height, int kernelSize) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    int inputDims[4] = {1, 1, height, width};
    int outputDims[4] = {1, 1, height - kernelSize + 1, width - kernelSize + 1};
    int kernelDims[4] = {1, 1, kernelSize, kernelSize};

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputDims[0], inputDims[1], inputDims[2], inputDims[3]);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputDims[0], outputDims[1], outputDims[2], outputDims[3]);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelDims[0], kernelDims[1], kernelDims[2], kernelDims[3]);

    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    float *d_input, *d_kernel, *d_output;
    size_t inputSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);
    size_t outputSize = (width - kernelSize + 1) * (height - kernelSize + 1) * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    cudnnConvolutionFwdAlgoPerf_t perfResults;
    int returnedAlgoCount;
    cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, 1, &returnedAlgoCount, &perfResults);

    cudnnConvolutionFwdAlgo_t algo = perfResults.algo;

    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize);

    void* d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_output);

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

