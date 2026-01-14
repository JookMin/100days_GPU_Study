#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_sub(const float* A, const float* B, float* C, int N) {
  const int x =  blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    C[x] = A[x] - B[x];
  } 
}

int main(){
  const int elementsNum = 1024;
  const int size = elementsNum * sizeof(float);

  float *h_x = (float*)malloc(size);
  float *h_y = (float*)malloc(size);
  float *h_z = (float*)malloc(size);

  for (int i=0; i<elementsNum; i++){
    h_x[i]=i;
    h_y[i]=i;
  }

  float *d_x,*d_y,*d_z;

  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_y, size);
  cudaMalloc((void**)&d_z, size);

  cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);

  dim3 blockDim(256,1,1);
  dim3 gridDim((elementsNum+blockDim.x-1)/blockDim.x,1,1);

  vector_sub<<<gridDim,blockDim>>>(d_x,d_y,d_z,elementsNum);

  cudaDeviceSynchronize();

  cudaMemcpy(h_z,d_z,size,cudaMemcpyDeviceToHost);

  for (int i =0; i < elementsNum; i++){
    printf("%f\n",h_z[i]);
  }

  //clean 
  free(h_x);
  free(h_y);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
}