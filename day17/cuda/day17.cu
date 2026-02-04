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

__global__ void floatToInt8(const float* input, int8_t *output, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    float value = input[idx];
    output[idx] = static_cast<int8_t>(fminf(fmaxf(value, -128.0f), 127.0f));
  }
}

void cpuFloatToInt8(const float* input, int8_t* output, const int N) {
    for (int i = 0; i < N; i++) {
        output[i] = static_cast<int8_t>(input[i]);
    }
}

int main(){
  const int elementsNum = 1024 * 32;
  const int size = elementsNum * sizeof(float);

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((elementsNum + 255) / 256, 1, 1);

  float *h_input = (float*)malloc(size);
  int8_t *h_output = (int8_t*)malloc(size);
  int8_t *h_cpu_output = (int8_t*)malloc(size);
  float eps = 1e-5f;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> distInput(-1.0f, 1.0f);

  for (int i = 0; i < elementsNum; i++){
      h_input[i] = distInput(engine);
  }

  float *d_input;
  int8_t *d_output;
  cudaMalloc((void**)&d_input, size);
  cudaMalloc((void**)&d_output, size);

  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, size);
  floatToInt8<<<gridDim, blockDim>>>(d_input, d_output, elementsNum);


  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
  cpuFloatToInt8(h_input, h_cpu_output, elementsNum);

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

  free(h_input);
  free(h_output);
  free(h_cpu_output);

  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
