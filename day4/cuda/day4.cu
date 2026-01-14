#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>

__global__ void fusedAXPBY(const float* X, const float* Y, float* Z, float a, float b, int N) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    Z[x] = a * X[x] + b * Y[x];
  } 
}

int main(){
  const int elementsNum = 1024;
  const int size = elementsNum * sizeof(float);

  float *h_x = (float*)malloc(size);
  float *h_y = (float*)malloc(size);
  float *h_z = (float*)malloc(size);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> dist(0.0f, 10.0f); // 0.0 ~ 10.0 사이 랜덤

  float a = dist(engine);
  float b = dist(engine);

  printf("Scalar a: %f, b: %f\n", a, b);

  for (int i = 0; i < elementsNum; i++){
    h_x[i] = dist(engine);
    h_y[i] = dist(engine);
  }

  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_y, size);
  cudaMalloc((void**)&d_z, size);

  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elementsNum + blockDim.x - 1) / blockDim.x, 1, 1);

  fusedAXPBY<<<gridDim, blockDim>>>(d_x, d_y, d_z, a, b, elementsNum);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

  int errorCount = 0;
  const float epsilon = 1e-4f; 

  for (int i = 0; i < elementsNum; i++){
    float cpu_result = a * h_x[i] + b * h_y[i];
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
  free(h_y);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
}
