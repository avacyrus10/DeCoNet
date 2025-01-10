#include <cudnn.h>
#include <iostream>
#include "correlation3d.h"

#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void correlate3D(const float* input, const float* kernel, float* output, int width, int height, int depth, int kernelSize) {
    std::cout << "Starting 3D correlation..." << std::endl;

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    int n = 1, c = 1;
    int outputWidth = width - kernelSize + 1;
    int outputHeight = height - kernelSize + 1;
    int outputDepth = depth - kernelSize + 1;

    // Define strides
    int inputStrides[5] = {c * depth * height * width, depth * height * width, height * width, width, 1};
    int outputStrides[5] = {c * outputDepth * outputHeight * outputWidth, outputDepth * outputHeight * outputWidth, outputHeight * outputWidth, outputWidth, 1};
    int kernelDims[5] = {c, c, kernelSize, kernelSize, kernelSize};

    // Set tensor descriptors
    int inputDims[5] = {n, c, depth, height, width};
    int outputDims[5] = {n, c, outputDepth, outputHeight, outputWidth};

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 5, inputDims, inputStrides));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 5, outputDims, outputStrides));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, kernelDims));

    // Set convolution descriptor
    int convPadding[3] = {0, 0, 0};
    int convStride[3] = {1, 1, 1};
    int convDilation[3] = {1, 1, 1};
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(convDesc, 3, convPadding, convStride, convDilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Select convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int returnedAlgoCount;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, 1, &returnedAlgoCount, &algoPerf));

    // Workspace size
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algoPerf.algo, &workspaceSize));
    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Allocate memory
    float* d_input;
    float* d_kernel;
    float* d_output;

    size_t inputSize = n * c * depth * height * width * sizeof(float);
    size_t kernelSizeBytes = c * kernelSize * kernelSize * kernelSize * sizeof(float);
    size_t outputSize = n * c * outputDepth * outputHeight * outputWidth * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelSizeBytes));
    CHECK_CUDA(cudaMalloc(&d_output, outputSize));

    CHECK_CUDA(cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice));

    // Perform correlation
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc, algoPerf.algo, d_workspace, workspaceSize, &beta, outputDesc, d_output));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernelDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    std::cout << "3D correlation completed successfully." << std::endl;
}

