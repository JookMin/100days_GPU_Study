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

__global__ void getMeanSquare(const float* input, float *d_mean, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    float val = input[idx];
    atomicAdd(d_mean, val * val / (float)N);
  }
}

__global__ void getOutput(const float* input, float *output, const float gamma, const float beta, int N, const float rms) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    output[idx] = gamma * (input[idx] / rms) + beta;
  }
}

void RMSNorm(const float* input, const float gamma, const float beta, float* output, int N, const float eps) {
  dim3 blockDim(256, 1, 1);
  dim3 gridDim((N + 255) / 256, 1, 1);

  float *d_mean;
  cudaMalloc((void**)&d_mean, sizeof(float));
  cudaMemset(d_mean, 0, sizeof(float));
  getMeanSquare<<<gridDim, blockDim>>>(input, d_mean, N);
  float h_mean;
  cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
  float rms = sqrtf(h_mean + eps);
  getOutput<<<gridDim, blockDim>>>(input, output, gamma, beta, N, rms);
  cudaDeviceSynchronize();
  cudaFree(d_mean);
}

void cpuRMSNorm(const float* input, const float gamma, const float beta, float* output, const int N, const float eps) {
  double mean = 0.0f;
  for (int i = 0; i < N; i++) {
    float val = input[i];
    mean += val * val;
  }
  float mean_sq = (float)(mean / N);
  float rms = std::sqrt(mean_sq + eps);
  
  for (int i = 0; i < N; i++) {
    float val = input[i];
    output[i] = (val/ rms) * gamma + beta;
  }
}

int main(){
  const int elementsNum = 1024 * 32;
  const int size = elementsNum * sizeof(float);

  float *h_input = (float*)malloc(size);
  float *h_output = (float*)malloc(size); 
  float *h_cpu_output = (float*)malloc(size);

  float gamma;
  float beta;
  float eps = 1e-5f; 

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> distInput(-100.0f, 100.0f);
  std::uniform_real_distribution<float> distGamma(0.1f, 10.0f);
  std::uniform_real_distribution<float> distBeta(-10.0f, 10.0f);

  for (int i = 0; i < elementsNum; i++){
      h_input[i] = distInput(engine);
  }

  gamma = distGamma(engine);
  beta = distBeta(engine);

  float *d_input, *d_output;
  cudaMalloc((void**)&d_input, size);
  cudaMalloc((void**)&d_output, size);

  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  RMSNorm(d_input, gamma, beta, d_output, elementsNum, eps);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
  cpuRMSNorm(h_input, gamma, beta, h_cpu_output, elementsNum, eps);

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
  
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
