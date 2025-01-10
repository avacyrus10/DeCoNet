#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "sma.h"

__global__ void smaKernel(const float* __restrict__ d_valA, const int* __restrict__ d_rowPtrA,
                          const int* __restrict__ d_colIdxA, const float* __restrict__ d_valB, 
                          const int* __restrict__ d_rowPtrB, const int* __restrict__ d_colIdxB, 
                          float* d_valC, int* d_rowPtrC, int* d_colIdxC, 
                          int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int idxC = d_rowPtrC[row];
        int idxA = d_rowPtrA[row];
        int idxB = d_rowPtrB[row];

        while (idxA < d_rowPtrA[row + 1] && idxB < d_rowPtrB[row + 1]) {
            int colA = d_colIdxA[idxA];
            int colB = d_colIdxB[idxB];

            if (colA == colB) {
                d_valC[idxC] = d_valA[idxA] + d_valB[idxB];
                d_colIdxC[idxC] = colA;
                idxA++;
                idxB++;
            } else if (colA < colB) {
                d_valC[idxC] = d_valA[idxA];
                d_colIdxC[idxC] = colA;
                idxA++;
            } else {
                d_valC[idxC] = d_valB[idxB];
                d_colIdxC[idxC] = colB;
                idxB++;
            }
            idxC++;
        }

        while (idxA < d_rowPtrA[row + 1]) {
            d_valC[idxC] = d_valA[idxA];
            d_colIdxC[idxC] = d_colIdxA[idxA];
            idxA++;
            idxC++;
        }

        while (idxB < d_rowPtrB[row + 1]) {
            d_valC[idxC] = d_valB[idxB];
            d_colIdxC[idxC] = d_colIdxB[idxB];
            idxB++;
            idxC++;
        }
    }
}

void sma(const std::vector<float> &h_valA, const std::vector<int> &h_rowPtrA, const std::vector<int> &h_colIdxA,
         const std::vector<float> &h_valB, const std::vector<int> &h_rowPtrB, const std::vector<int> &h_colIdxB,
         std::vector<float> &h_valC, std::vector<int> &h_rowPtrC, std::vector<int> &h_colIdxC, int num_rows) {
    float *d_valA, *d_valB, *d_valC;
    int *d_rowPtrA, *d_colIdxA, *d_rowPtrB, *d_colIdxB, *d_rowPtrC, *d_colIdxC;

    int nnzA = h_valA.size();
    int nnzB = h_valB.size();
    int nnzC = h_valC.size();

    cudaMalloc(&d_valA, nnzA * sizeof(float));
    cudaMalloc(&d_rowPtrA, h_rowPtrA.size() * sizeof(int));
    cudaMalloc(&d_colIdxA, h_colIdxA.size() * sizeof(int));
    cudaMalloc(&d_valB, nnzB * sizeof(float));
    cudaMalloc(&d_rowPtrB, h_rowPtrB.size() * sizeof(int));
    cudaMalloc(&d_colIdxB, h_colIdxB.size() * sizeof(int));
    cudaMalloc(&d_valC, nnzC * sizeof(float));
    cudaMalloc(&d_rowPtrC, h_rowPtrC.size() * sizeof(int));
    cudaMalloc(&d_colIdxC, h_colIdxC.size() * sizeof(int));

    cudaMemcpy(d_valA, h_valA.data(), nnzA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrA, h_rowPtrA.data(), h_rowPtrA.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxA, h_colIdxA.data(), h_colIdxA.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valB, h_valB.data(), nnzB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrB, h_rowPtrB.data(), h_rowPtrB.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxB, h_colIdxB.data(), h_colIdxB.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrC, h_rowPtrC.data(), h_rowPtrC.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    smaKernel<<<numBlocks, blockSize>>>(d_valA, d_rowPtrA, d_colIdxA, d_valB, d_rowPtrB, d_colIdxB, d_valC, d_rowPtrC, d_colIdxC, num_rows);

    cudaMemcpy(h_valC.data(), d_valC, nnzC * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdxC.data(), d_colIdxC, h_colIdxC.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_valA);
    cudaFree(d_rowPtrA);
    cudaFree(d_colIdxA);
    cudaFree(d_valB);
    cudaFree(d_rowPtrB);
    cudaFree(d_colIdxB);
    cudaFree(d_valC);
    cudaFree(d_rowPtrC);
    cudaFree(d_colIdxC);
}

