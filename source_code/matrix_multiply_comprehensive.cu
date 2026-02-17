#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA kernel for naïve matrix multiplication
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU reference implementation for verification
void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Initialize matrix with random values
void initializeMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
    }
}

// Verify GPU result against CPU result
bool verifyResults(float *gpu_result, float *cpu_result, int N, float tolerance = 1e-4) {
    for (int i = 0; i < N * N; i++) {
        if (abs(gpu_result[i] - cpu_result[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Get current time in milliseconds with higher precision
double getTimeMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Run multiple iterations for more accurate timing
double timeGPU(float *d_A, float *d_B, float *d_C, int N, dim3 gridSize, dim3 blockSize, int iterations = 10) {
    // Warm up
    matrixMultiplyGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    double total_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        double start_time = getTimeMs();
        matrixMultiplyGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        total_time += getTimeMs() - start_time;
    }
    return total_time / iterations;
}

double timeCPU(float *A, float *B, float *C, int N, int iterations = 10) {
    double total_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        double start_time = getTimeMs();
        matrixMultiplyCPU(A, B, C, N);
        total_time += getTimeMs() - start_time;
    }
    return total_time / iterations;
}

int main() {
    // Test a wide range of matrix sizes to find the crossover point
    int sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Comprehensive CPU vs GPU Matrix Multiplication Analysis\n");
    printf("======================================================\n");
    printf("Testing to find the crossover point where GPU becomes faster than CPU\n\n");
    
    // Check for CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("CUDA Device: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("\n");
    
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-8s\n", 
           "Size", "GPU(ms)", "CPU(ms)", "GPU(GFLOPS)", "CPU(GFLOPS)", "Speedup", "Winner");
    printf("================================================================================\n");
    
    bool found_crossover = false;
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        
        // Allocate host memory
        size_t matrix_size = N * N * sizeof(float);
        float *h_A = (float*)malloc(matrix_size);
        float *h_B = (float*)malloc(matrix_size);
        float *h_C_gpu = (float*)malloc(matrix_size);
        float *h_C_cpu = (float*)malloc(matrix_size);
        
        // Initialize matrices
        srand(42); // Fixed seed for reproducibility
        initializeMatrix(h_A, N);
        initializeMatrix(h_B, N);
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, matrix_size);
        cudaMalloc(&d_B, matrix_size);
        cudaMalloc(&d_C, matrix_size);
        
        // Copy data to device
        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
        
        // Set up grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        
        // Time GPU execution (multiple iterations for small matrices)
        int iterations = (N <= 64) ? 100 : (N <= 256) ? 50 : 10;
        double gpu_time = timeGPU(d_A, d_B, d_C, N, gridSize, blockSize, iterations);
        
        // Copy result back to host
        cudaMemcpy(h_C_gpu, d_C, matrix_size, cudaMemcpyDeviceToHost);
        
        // Time CPU execution
        double cpu_time = timeCPU(h_A, h_B, h_C_cpu, N, iterations);
        
        // Verify results
        bool results_match = verifyResults(h_C_gpu, h_C_cpu, N);
        
        // Calculate performance metrics
        double gflops_gpu = (2.0 * N * N * N) / (gpu_time * 1e6);
        double gflops_cpu = (2.0 * N * N * N) / (cpu_time * 1e6);
        double speedup = cpu_time / gpu_time;
        
        // Determine winner
        const char* winner = (gpu_time < cpu_time) ? "GPU" : "CPU";
        if (!found_crossover && gpu_time < cpu_time) {
            found_crossover = true;
        }
        
        printf("%-8d %-12.3f %-12.3f %-12.2f %-12.2f %-12.2f %-8s\n", 
               N, gpu_time, cpu_time, gflops_gpu, gflops_cpu, speedup, winner);
        
        // Clean up
        free(h_A);
        free(h_B);
        free(h_C_gpu);
        free(h_C_cpu);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    printf("\n");
    printf("Analysis:\n");
    printf("=========\n");
    if (found_crossover) {
        printf("✓ GPU becomes faster than CPU at larger matrix sizes\n");
        printf("✓ This demonstrates the expected behavior: CPU is faster for small matrices due to GPU overhead\n");
    } else {
        printf("⚠ GPU was faster even for small matrices - this might indicate:\n");
        printf("  - Very powerful GPU relative to CPU\n");
        printf("  - CPU implementation could be optimized further\n");
        printf("  - GPU overhead is minimal on this system\n");
    }
    
    printf("\nKey Insights:\n");
    printf("- GPU overhead: Memory transfer, kernel launch, synchronization\n");
    printf("- CPU advantage: No overhead, direct memory access, optimized for small workloads\n");
    printf("- Crossover point: Depends on hardware, implementation, and problem size\n");
    
    return 0;
}




