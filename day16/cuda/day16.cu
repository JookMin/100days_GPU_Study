#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

__global__ void Int8QuantMatMulKernel(const int8_t* A, const int8_t* B, int8_t* C, 
                                      int M, int N, int K, 
                                      float scale, int zpA, int zpB, int zpC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            int32_t a_val = static_cast<int32_t>(A[row * K + k]) - zpA;
            int32_t b_val = static_cast<int32_t>(B[k * N + col]) - zpB;
            sum += a_val * b_val;
        }

        float res = (float)sum * scale + (float)zpC;
        
        int32_t quantized = (int32_t)roundf(res);

        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;

        C[row * N + col] = static_cast<int8_t>(quantized);
    }
}

void Int8QuantMatMul(const int8_t* d_A, const int8_t* d_B, int8_t* d_C, 
                     int M, int N, int K, 
                     float scaleA, float scaleB, float scaleOut, 
                     int zpA, int zpB, int zpC) {
    
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, 1);

    float effectiveScale = scaleA * scaleB / scaleOut;

    Int8QuantMatMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, effectiveScale, zpA, zpB, zpC);
    cudaDeviceSynchronize();
}

void cpuInt8QuantMatMul(const int8_t* A, const int8_t* B, int8_t* C, 
                        int M, int N, int K, 
                        float scaleA, float scaleB, float scaleOut, 
                        int zpA, int zpB, int zpC) {
    
    float scale = scaleA * scaleB / scaleOut;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                int32_t a_val = static_cast<int32_t>(A[m * K + k]) - zpA;
                int32_t b_val = static_cast<int32_t>(B[k * N + n]) - zpB;
                sum += a_val * b_val;
            }
            
            float res = (float)sum * scale + (float)zpC;
            int32_t quantized = static_cast<int32_t>(std::round(res));
            
            quantized = std::max(-128, std::min(127, quantized));
            C[m * N + n] = static_cast<int8_t>(quantized);
        }
    }
}

int main(){
    const int M = 64;
    const int N = 64;
    const int K = 128;

    const int sizeA = M * K * sizeof(int8_t);
    const int sizeB = K * N * sizeof(int8_t);
    const int sizeC = M * N * sizeof(int8_t);

    float scaleA = 0.02f;
    float scaleB = 0.02f;
    float scaleOut = 0.05f;
    int zpA = 0;
    int zpB = 0;
    int zpC = 0;

    int8_t *h_A = (int8_t*)malloc(sizeA);
    int8_t *h_B = (int8_t*)malloc(sizeB);
    int8_t *h_output_gpu = (int8_t*)malloc(sizeC); 
    int8_t *h_output_cpu = (int8_t*)malloc(sizeC);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> distVal(-128, 127);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<int8_t>(distVal(engine));
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<int8_t>(distVal(engine));

    int8_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC);

    Int8QuantMatMul(d_A, d_B, d_C, M, N, K, scaleA, scaleB, scaleOut, zpA, zpB, zpC);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_output_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

    cpuInt8QuantMatMul(h_A, h_B, h_output_cpu, M, N, K, scaleA, scaleB, scaleOut, zpA, zpB, zpC);

    int errorCount = 0;
    int elementsNum = M * N;

    for (int i = 0; i < elementsNum; i++) {
        if (h_output_gpu[i] != h_output_cpu[i]) {
            errorCount++;
            if (errorCount < 10) {
                printf("Mismatch at index %d: GPU=%d, CPU=%d\n", i, h_output_gpu[i], h_output_cpu[i]);
            }
        }
    }

    if (errorCount == 0) {                          
        printf("✅ 검증 성공! (All %d elements match)\n", elementsNum);
    } else {
        printf("❌ 검증 실패! (Found %d mismatches)\n", errorCount);
    }

    free(h_A);
    free(h_B);
    free(h_output_gpu);
    free(h_output_cpu);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}