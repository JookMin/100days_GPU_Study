#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>

__global__ void LeakyReLu(const float* X, float* Z, float a, int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    Z[x] = X[x] > 0 ? X[x] : X[x] * a;
  } 
}

int main(){
  const int elementsNum = 1024;
  const int size = elementsNum * sizeof(float);

  float *h_x = (float*)malloc(size);
  float *h_z = (float*)malloc(size);

  float a;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);

  for (int i = 0; i < elementsNum; i++){
    h_x[i] = dist(engine);
  }

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 0.0f);
  a = dist(engine);

  float *d_x, *d_z;
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_z, size);

  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elementsNum + blockDim.x - 1) / blockDim.x, 1, 1);

  LeakyReLu<<<gridDim, blockDim>>>(d_x, d_z, a, elementsNum);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

  int errorCount = 0;
  const float epsilon = 1e-4f; 

  for (int i = 0; i < elementsNum; i++){
    float cpu_result = h_x[i] > 0 ? h_x[i]: h_x[i] * a;
    if (std::fabs(cpu_result - h_z[i]) > epsilon) {
      errorCount++;
    }
  }

  if (errorCount == 0) {
    printf("✅ 검증 성공! (All %d elements match)\n", elementsNum);
  } else {
    printf("❌ 검증 실패! (Found %d mismatches)\n", errorCount);
  }

  free(h_x);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_z);

  return 0;
}
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>

__global__ void LeakyReLu(const float* X, float* Z, float a, int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    Z[x] = X[x] > 0 ? X[x] : X[x] * a;
  } 
}

int main(){
  const int elementsNum = 1024;
  const int size = elementsNum * sizeof(float);

  float *h_x = (float*)malloc(size);
  float *h_z = (float*)malloc(size);

  float a;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine1(seed);
  std::uniform_real_distribution<float> dist1(0.0f, 10.0f);

  for (int i = 0; i < elementsNum; i++){
    h_x[i] = dist1(engine1);
  }

  std::mt19937 engine2(seed);
  std::uniform_real_distribution<float> dist2(-1.0f, 0.0f);
  a = dist2(engine2);

  float *d_x, *d_z;
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_z, size);

  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elementsNum + blockDim.x - 1) / blockDim.x, 1, 1);

  LeakyReLu<<<gridDim, blockDim>>>(d_x, d_z, a, elementsNum);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

  int errorCount = 0;
  const float epsilon = 1e-4f; 

  for (int i = 0; i < elementsNum; i++){
    float cpu_result = h_x[i] > 0 ? h_x[i]: h_x[i] * a;
    if (std::fabs(cpu_result - h_z[i]) > epsilon) {
      errorCount++;
    }
  }

  if (errorCount == 0) {
    printf("✅ 검증 성공! (All %d elements match)\n", elementsNum);
  } else {
    printf("❌ 검증 실패! (Found %d mismatches)\n", errorCount);
  }

  free(h_x);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_z);

  return 0;
}
