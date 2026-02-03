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

__global__ void Dot(const float* A, const float *B, float *output, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    float input = A[idx];
    float weight = B[idx];
    float mul = input * weight;
    atomicAdd(&output[0], mul);
  }
}

void cpuDot(const float* A, const float* B, float* output, const int N) {
  for (int i = 0; i < N; i++) {
    output[i] = A[i] * B[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < N; i ++) {
    sum += output[i];
  }
  output[0] = sum;
}

int main(){
  const int elementsNum = 1024 * 32;
  const int size = elementsNum * sizeof(float);

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elementsNum + 255) / 256, 1, 1);

  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_output = (float*)malloc(size); 
  float *h_cpu_output = (float*)malloc(size);
  float eps = 1e-5f; 

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> distInputA(-1.0f, 1.0f);
  std::uniform_real_distribution<float> distInputB(-1.0f, 1.0f);

  for (int i = 0; i < elementsNum; i++){
      h_A[i] = distInputA(engine);
      h_B[i] = distInputB(engine);
  }

  float *d_A, *d_B, *d_output;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_output, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, size);
  Dot<<<gridDim, blockDim>>>(d_A, d_B, d_output, elementsNum);


  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
  cpuDot(h_A, h_B, h_cpu_output, elementsNum);

  int errorCount = 0;

  float diff = std::fabs(h_output[0] - h_cpu_output[0]);
  if (diff > eps) {
    errorCount++;
  }

  if (errorCount == 0) {                          
    printf("✅ 검증 성공! (All %d elements match)\n", elementsNum);
  } else {
    printf("❌ 검증 실패! (Found %d mismatches)\n", errorCount);
  }

  free(h_A);
  free(h_B);
  free(h_output);
  free(h_cpu_output);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_output);

  return 0;
}
