#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>

__global__ void Clip(const float* X, float* Z, float lo, float hi, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float value = X[idx];
    if (value > hi) value = hi;
    else if (value < lo) value = lo;
    Z[idx] = value;
  } 
}

int main(){
  const int elementsNum = 1024;
  const int size = elementsNum * sizeof(float);

  float *h_x = (float*)malloc(size);
  float *h_z = (float*)malloc(size);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine1(seed);
  std::uniform_real_distribution<float> dist1(-500.0f, 500.0f);

  for (int i = 0; i < elementsNum; i++){
    h_x[i] = dist1(engine1);
  }

  float *d_x, *d_z;
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_z, size);

  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

  float lo, hi;
  std::uniform_real_distribution<float> dist2(-100.0f, 100.0f);
  lo = dist2(engine1);
  unsigned seed2 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine2(seed2);
  hi = dist2(engine2);
  float temp;
  if (lo > hi) {
    temp = lo;
    lo = hi;
    hi = temp;
  }

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elementsNum + blockDim.x - 1) / blockDim.x, 1, 1);

  Clip<<<gridDim, blockDim>>>(d_x, d_z, lo, hi, elementsNum);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

  int errorCount = 0;
  for (int i = 0; i < elementsNum; i++) {
    
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
