// matrix_lib.cu
#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#define CUDA_OK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return; \
  } \
} while(0)

__global__ void matrixMultiplyTiled(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x, ty = threadIdx.y;
  int Row = blockIdx.y * TILE_WIDTH + ty;
  int Col = blockIdx.x * TILE_WIDTH + tx;

  float Pvalue = 0.0f;
  int tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int m = 0; m < tiles; ++m) {
    int aCol = m * TILE_WIDTH + tx;
    int bRow = m * TILE_WIDTH + ty;

    ds_A[ty][tx] = (Row < N && aCol < N) ? A[Row * N + aCol] : 0.0f;
    ds_B[ty][tx] = (bRow < N && Col < N) ? B[bRow * N + Col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += ds_A[ty][k] * ds_B[k][tx];
    __syncthreads();
  }
  if (Row < N && Col < N) C[Row * N + Col] = Pvalue;
}

extern "C" void gpu_matrix_multiply(const float *h_A,
                                    const float *h_B,
                                    float *h_C,
                                    int N) {
  size_t bytes = (size_t)N * N * sizeof(float);
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  CUDA_OK(cudaMalloc(&d_A, bytes));
  CUDA_OK(cudaMalloc(&d_B, bytes));
  CUDA_OK(cudaMalloc(&d_C, bytes));

  CUDA_OK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

  matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
  cudaError_t kerr = cudaGetLastError();
  if (kerr != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kerr));
  }
  CUDA_OK(cudaDeviceSynchronize());

  CUDA_OK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

