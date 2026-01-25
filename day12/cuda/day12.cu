#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm> 

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        
        float assumed_float = *(float*)&assumed;
        float max_val = fmaxf(val, assumed_float);
        int max_val_int = *(int*)&max_val;
        old = atomicCAS(address_as_int, assumed, max_val_int);

    } while (assumed != old);

    return *(float*)&old;
}

__global__ void normalize_kernel(float* output, const float *sum, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output[idx] = output[idx] / (*sum);
    }
}

__global__ void exp_sum_kernel(const float* input, float* output, float* sum, const float *max, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        float inputEXP = expf(input[idx] - *max);
        output[idx] = inputEXP;
        atomicAdd(sum, inputEXP);
    }
}

__global__ void find_max_kernel(const float* input, float* max, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        float inputVal = input[idx];
        atomicMaxFloat(max, inputVal);
    }
}

void softMax(const float* input, float* output, int N) {
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 1, 1);

    float* d_global_sum;
    float* d_global_max;
    cudaMalloc((void**)&d_global_sum, sizeof(float));
    cudaMalloc((void**)&d_global_max, sizeof(float));

    cudaMemset(d_global_sum, 0, sizeof(float));
    
    float min_float = -FLT_MAX;
    cudaMemcpy(d_global_max, &min_float, sizeof(float), cudaMemcpyHostToDevice);

    find_max_kernel<<<gridDim, blockDim>>>(input, d_global_max, N);
    exp_sum_kernel<<<gridDim, blockDim>>>(input, output, d_global_sum, d_global_max, N);
    normalize_kernel<<<gridDim, blockDim>>>(output, d_global_sum, N);
    
    cudaDeviceSynchronize();

    cudaFree(d_global_sum);
    cudaFree(d_global_max);
}

void cpuSoftmax(const float* input, float* output, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

int main(){
    const int elementsNum = 1024;
    const int size = elementsNum * sizeof(float);
  
    float *h_x = (float*)malloc(size);
    float *h_z = (float*)malloc(size); 
    float *h_cpu_z = (float*)malloc(size);
  
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
  
    softMax(d_x, d_z, elementsNum);
  
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
  
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
  
    cpuSoftmax(h_x, h_cpu_z, elementsNum);

    int errorCount = 0;
    const float epsilon = 1e-5f; 

    for (int i = 0; i < elementsNum; i++) {
        float diff = std::fabs(h_z[i] - h_cpu_z[i]);
        if (diff > epsilon) {
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
    free(h_cpu_z);
    cudaFree(d_x);
    cudaFree(d_z);
  
    return 0;
}                                                                                                                                   