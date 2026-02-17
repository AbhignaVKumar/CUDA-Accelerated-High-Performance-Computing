// conv_cpu.c - CPU implementation of 2D convolution for comparison
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

void conv2d_cpu_u32(const uint32_t* img, int H, int W,
                    const int32_t* ker, int K,
                    int32_t* out) {
    int pad = K / 2;
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int64_t acc = 0;
            
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    int img_y = y + ky - pad;
                    int img_x = x + kx - pad;
                    
                    uint32_t img_val = 0;
                    if (img_y >= 0 && img_y < H && img_x >= 0 && img_x < W) {
                        img_val = img[img_y * W + img_x];
                    }
                    
                    acc += (int64_t)img_val * (int64_t)ker[ky * K + kx];
                }
            }
            
            out[y * W + x] = (int32_t)acc;
        }
    }
}

// Test function
int main() {
    // Test with different sizes
    int sizes[] = {256, 512, 1024};
    int kernel_sizes[] = {3, 5, 7};
    
    for (int s = 0; s < 3; s++) {
        int H = sizes[s];
        int W = sizes[s];
        
        for (int k = 0; k < 3; k++) {
            int K = kernel_sizes[k];
            
            // Allocate memory
            size_t img_size = H * W * sizeof(uint32_t);
            size_t ker_size = K * K * sizeof(int32_t);
            size_t out_size = H * W * sizeof(int32_t);
            
            uint32_t* img = (uint32_t*)malloc(img_size);
            int32_t* ker = (int32_t*)malloc(ker_size);
            int32_t* out = (int32_t*)malloc(out_size);
            
            // Initialize with random data
            for (int i = 0; i < H * W; i++) {
                img[i] = rand() % 256;
            }
            
            for (int i = 0; i < K * K; i++) {
                ker[i] = 1; // Simple box filter
            }
            
            // Benchmark
            clock_t start = clock();
            conv2d_cpu_u32(img, H, W, ker, K, out);
            clock_t end = clock();
            
            double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
            
            printf("CPU: %dx%d image, %dx%d kernel: %.6f seconds\n", 
                   H, W, K, K, cpu_time);
            
            free(img);
            free(ker);
            free(out);
        }
    }
    
    return 0;
}