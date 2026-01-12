#include <cuda_runtime.h>

__global__ void vector_scalar(const float* A, const float* B, float alpha, int N) {
  const int x =  blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    B[x] = A[x] * alpha;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float alpha, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, alpha, N);
    cudaDeviceSynchronize();
}
