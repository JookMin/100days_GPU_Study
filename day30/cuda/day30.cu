#include <cuda_runtime.h>

__global__ void nomalize_kernel(float* input, const float *sum, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        input[idx] = input[idx] / (*sum);
    }
}

__global__ void exp_sum_kernel(const float* input, float* output, float* sum, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        float inputEXP = expf(input[idx]);
        output[idx] = inputEXP;
        atomicAdd(sum, inputEXP);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_global_sum;
    cudaMalloc((void**)&d_global_sum, sizeof(float));
    cudaMemset(d_global_sum, 0, sizeof(float));

    exp_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_global_sum, N);
    nomalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, d_global_sum, N);
    cudaDeviceSynchronize();
}
