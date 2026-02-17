// conv_cuda.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#ifndef TILE
#define TILE 16
#endif
// K <= 31 assumed; pass K at runtime. Use shared memory with halo.
__global__ void conv2d_u32_kernel(const uint32_t* __restrict__ img, int H, int W,
                                  const int32_t* __restrict__ ker, int K,
                                  int32_t* __restrict__ out) {
  extern __shared__ uint32_t tile[]; // size: (TILE+K-1)*(TILE+K-1)
  int r = K/2;

  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * TILE + tx;
  int y = blockIdx.y * TILE + ty;

  int shW = TILE + K - 1;
  int shx = tx, shy = ty;

  // Load with halo. Each thread may need to loop if shmem > block size.
  for (int dy = 0; dy < shW; dy += blockDim.y) {
    for (int dx = 0; dx < shW; dx += blockDim.x) {
      int lx = shx + dx;
      int ly = shy + dy;
      if (ly < shW && lx < shW) {
        int gx = blockIdx.x * TILE + lx - r;
        int gy = blockIdx.y * TILE + ly - r;
        uint32_t v = 0;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) v = img[gy*W + gx];
        tile[ly*shW + lx] = v;
      }
    }
  }
  __syncthreads();

  if (x < W && y < H) {
    int64_t acc = 0;
    for (int ky=0; ky<K; ++ky)
      for (int kx=0; kx<K; ++kx)
        acc += (int64_t)tile[(ty+ky)*shW + (tx+kx)] * (int64_t)ker[ky*K + kx];
    out[y*W + x] = (int32_t)acc;
  }
}

// Exposed C ABI for ctypes
extern "C" void conv2d_cuda_u32(const uint32_t* h_img, int H, int W,
                                const int32_t* h_ker, int K,
                                int32_t* h_out) {
  size_t ib = (size_t)H*W*sizeof(uint32_t);
  size_t kb = (size_t)K*K*sizeof(int32_t);
  size_t ob = (size_t)H*W*sizeof(int32_t);

  uint32_t *d_img=nullptr; int32_t *d_ker=nullptr, *d_out=nullptr;
  cudaMalloc(&d_img, ib); cudaMalloc(&d_ker, kb); cudaMalloc(&d_out, ob);
  cudaMemcpy(d_img, h_img, ib, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ker, h_ker, kb, cudaMemcpyHostToDevice);

  dim3 block(TILE, TILE);
  dim3 grid((W + TILE - 1)/TILE, (H + TILE - 1)/TILE);
  size_t sh = (size_t)(TILE + K - 1)*(TILE + K - 1)*sizeof(uint32_t);

  conv2d_u32_kernel<<<grid, block, sh>>>(d_img, H, W, d_ker, K, d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, ob, cudaMemcpyDeviceToHost);
  cudaFree(d_img); cudaFree(d_ker); cudaFree(d_out);
}
