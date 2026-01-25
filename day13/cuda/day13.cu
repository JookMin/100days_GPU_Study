#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm> 

#include <cuda_runtime.h>

__global__ void getMean(const float* input, float *mean, int N, int C) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (n < N && c < C) {
        int idx = n * C + c;
        atomicAdd(&mean[c], input[idx] / (float)N);
    }
}

__global__ void getSigma(const float* input, float *mean, float *sigma, int N, int C) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (c < C && n < N) {
        int idx = n * C + c;
        float gap = input[idx] - mean[c];
        atomicAdd(&sigma[c], (gap * gap) / (float)N);
    }
}

__global__ void getXHat(const float* input, float* output, float *mean, float* sigma, float eps, int N, int C) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (c < C && n < N) {
        int idx = n * C + c;
        float gap = input[idx] - mean[c];
        float under = sqrtf(sigma[c] + eps);
        output[idx] = gap / under;
    }
}

__global__ void getOutput(float* output, const float* gamma, const float *beta, int N, int C) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (c < C && n < N) {
        int idx = n * C + c;
        output[idx] = gamma[c] * output[idx] + beta[c];
    }
}

// input, gamma, beta, output are device pointers
void BatchNorm(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((N + 15) / 16, (C + 15) / 16, 1);

    float* d_mean;
    float* d_sigma;
    size_t size = C * sizeof(float);
    cudaMalloc((void**)&d_mean, size);
    cudaMalloc((void**)&d_sigma, size);
    cudaMemset(d_mean, 0, size);
    cudaMemset(d_sigma, 0, size);

    getMean<<<gridDim, blockDim>>>(input, d_mean, N, C);
    getSigma<<<gridDim, blockDim>>>(input, d_mean, d_sigma, N, C);
    getXHat<<<gridDim, blockDim>>>(input, output, d_mean, d_sigma, eps, N, C);
    getOutput<<<gridDim, blockDim>>>(output, gamma, beta, N, C);
    cudaDeviceSynchronize();

    cudaFree(d_mean);
    cudaFree(d_sigma);
}

void cpuBatchNorm(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    std::vector<float> mean(C, 0.0f);
    std::vector<float> var(C, 0.0f);

    for (int c = 0; c < C; c++) {
        for (int n = 0; n < N; n++) {
            mean[c] += input[n * C + c];
        }
        mean[c] /= N;
    }

    for (int c = 0; c < C; c++) {
        for (int n = 0; n < N; n++) {
            float diff = input[n * C + c] - mean[c];
            var[c] += diff * diff;
        }
        var[c] /= N;
    }

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            int idx = n * C + c;
            float std_dev = std::sqrt(var[c] + eps);
            float x_hat = (input[idx] - mean[c]) / std_dev;
            output[idx] = gamma[c] * x_hat + beta[c];
        }
    }
}

int main(){
    const int N = 32;
    const int C = 32;
    const int elementsNum = N * C;
    const int size = elementsNum * sizeof(float);
    const int sizeC = C * sizeof(float);
  
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size); 
    float *h_cpu_output = (float*)malloc(size);

    float *h_gamma = (float*)malloc(sizeC);
    float *h_beta = (float*)malloc(sizeC);
    float eps = 1e-5f; 
  
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> distInput(-100.0f, 100.0f);
    std::uniform_real_distribution<float> distGamma(0.1f, 10.0f);
    std::uniform_real_distribution<float> distBeta(-10.0f, 10.0f);
  
    for (int i = 0; i < elementsNum; i++){
        h_input[i] = distInput(engine);
    }

    for (int i = 0; i < C; i++) {
        h_gamma[i] = distGamma(engine);
        h_beta[i] = distBeta(engine);
    }
  
    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_gamma, sizeC);
    cudaMalloc((void**)&d_beta, sizeC);
  
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, sizeC, cudaMemcpyHostToDevice);
    BatchNorm(d_input, d_gamma, d_beta, d_output, N, C, eps);
  
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
  
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cpuBatchNorm(h_input, h_gamma, h_beta, h_cpu_output, N, C, eps);

    int errorCount = 0;

    for (int i = 0; i < elementsNum; i++) {
        float diff = std::fabs(h_output[i] - h_cpu_output[i]);
        if (diff > eps) {
            errorCount++;
        }
    }
  
    if (errorCount == 0) {                          
      printf("✅ 검증 성공! (All %d elements match)\n", elementsNum);
    } else {
      printf("❌ 검증 실패! (Found %d mismatches)\n", errorCount);
    }
  
    free(h_input);
    free(h_output);
    free(h_cpu_output);
    free(h_gamma);
    free(h_beta);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
  
    return 0;
}
