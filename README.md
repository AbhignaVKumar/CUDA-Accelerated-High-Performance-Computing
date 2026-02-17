# CUDA Accelerated High-Performance Computing on GCP

A comprehensive study of GPU-accelerated computing using CUDA, comparing CPU vs GPU performance for matrix operations and image processing. This project demonstrates the implementation and optimization of parallel algorithms on NVIDIA GPUs, achieving up to **18,000×** speedup over CPU implementations.

## 📋 Project Overview

This laboratory explores:
- **CPU vs GPU Performance Analysis** for matrix multiplication
- **CUDA Kernel Optimization** using shared memory tiling
- **cuBLAS Library Integration** for production-grade performance
- **Custom CUDA Libraries** for Python integration
- **2D Convolution** for image processing applications
- **Performance Scaling Analysis** across different problem sizes

---

## 🎯 Key Results

### Matrix Multiplication Performance (N=2048)

| Implementation | Time | GFLOPS | Speedup vs CPU |
|----------------|------|--------|----------------|
| **CPU (C)** | 52,961 ms | 0.35 | 1× |
| **Naïve CUDA** | 7.04 ms | 2,254 | 6,868× |
| **Tiled CUDA** | 5.77 ms | 3,014 | 8,514× |
| **cuBLAS** | 0.76 ms | - | **18,184×** |

### Convolution Performance (1024×1024 Images)

| Filter Type | CUDA Time | Speedup vs CPU |
|-------------|-----------|----------------|
| Sobel Edge Detection | 7.5 ms | ~1,872× |
| Gaussian Blur (5×5) | 2.5 ms | ~9,692× |
| Laplacian | 7.3 ms | ~1,872× |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Host (CPU)                        │
│  - Data Preparation                         │
│  - Memory Allocation                        │
│  - Kernel Launch                            │
└──────────────┬──────────────────────────────┘
               │ PCIe Transfer
               ▼
┌─────────────────────────────────────────────┐
│           Device (GPU)                      │
│  ┌─────────────────────────────────────┐   │
│  │  Global Memory (47 GB)              │   │
│  └──────────┬──────────────────────────┘   │
│             │                               │
│  ┌──────────▼──────────────────────────┐   │
│  │  Streaming Multiprocessors (SMs)    │   │
│  │  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ Shared Memory│  │ Shared Memory│ │   │
│  │  │   (48 KB)    │  │   (48 KB)    │ │   │
│  │  ├──────────────┤  ├──────────────┤ │   │
│  │  │Thread Blocks │  │Thread Blocks │ │   │
│  │  │  (16×16)     │  │  (16×16)     │ │   │
│  │  └──────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

**Hardware Requirements:**
- NVIDIA GPU with CUDA support (Compute Capability ≥ 3.5)
- Recommended: Tesla T4, V100, A6000, or similar

**Software Requirements:**
- Ubuntu 20.04 LTS or later
- NVIDIA Driver 470+
- CUDA Toolkit 11.0+
- Python 3.8+
- GCC 7.5+

### Installation

#### 1. Setup Google Cloud VM with GPU

```bash
# Create a GPU instance
gcloud compute instances create cuda-lab \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# SSH into the instance
gcloud compute ssh cuda-lab --zone=us-central1-a
```

#### 2. Install NVIDIA Drivers and CUDA Toolkit

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install NVIDIA driver
sudo apt-get install -y nvidia-driver-470

# Install CUDA Toolkit
sudo apt-get install -y nvidia-cuda-toolkit

# Verify installation
nvidia-smi
nvcc --version
```

#### 3. Install Python Dependencies

```bash
# Install Python packages
pip install numpy scipy pillow matplotlib

# For image processing
sudo apt-get install -y python3-opencv
```

---

## 💻 Project Structure

```
cuda-performance-lab/
├── README.md
├── src/
│   ├── cpu/
│   │   ├── matrix_cpu.c              # CPU matrix multiplication
│   │   ├── conv_cpu.c                # CPU 2D convolution
│   │   └── Makefile
│   ├── cuda/
│   │   ├── matrix_naive.cu           # Naïve CUDA kernel
│   │   ├── matrix_tiled.cu           # Optimized tiled kernel
│   │   ├── matrix_cublas.cu          # cuBLAS implementation
│   │   ├── conv_cuda.cu              # CUDA convolution
│   │   └── Makefile
│   ├── library/
│   │   ├── matrix_lib.cu             # Shared library for Python
│   │   ├── conv_lib.cu               # Convolution library
│   │   ├── libgpukit.so              # Compiled shared library
│   │   └── Makefile
│   └── python/
│       ├── py_matrix.py              # Python matrix multiplication
│       ├── py_conv.py                # Python convolution wrapper
│       └── benchmark.py              # Performance benchmarking
├── data/
│   ├── images/                       # Test images
│   └── filters/                      # Convolution kernels
├── results/
│   ├── performance_plots.png
│   └── processed_images/
└── scripts/
    ├── run_benchmarks.sh
    ├── setup_environment.sh
    └── generate_plots.py
```

---

## 🔧 Usage

### Part 1: CPU Matrix Multiplication

```bash
# Compile
gcc -O2 src/cpu/matrix_cpu.c -o matrix_cpu

# Run with different matrix sizes
./matrix_cpu 512
./matrix_cpu 1024
./matrix_cpu 2048
```

**Expected Output:**
```
CPU execution time (N=512): 0.368668 seconds
CPU execution time (N=1024): 2.993430 seconds
CPU execution time (N=2048): 25.564445 seconds
```

### Part 2: Naïve CUDA Implementation

```bash
# Compile
nvcc -O3 -arch=sm_86 src/cuda/matrix_naive.cu -o matrix_naive

# Run
./matrix_naive 1024
```

**Expected Output:**
```
GPU Time: 0.92 ms
CPU Time: 4576.29 ms
GPU Performance: 2335.90 GFLOPS
Speedup: 4977.82×
```

### Part 3: Optimized Tiled CUDA

```bash
# Compile with optimizations
nvcc -O3 -std=c++11 -arch=sm_86 src/cuda/matrix_tiled.cu -o matrix_tiled

# Run
./matrix_tiled 2048
```

**Key Features:**
- 16×16 tile size for optimal shared memory usage
- Coalesced global memory access
- Reduced memory transactions

### Part 4: cuBLAS Implementation

```bash
# Compile with cuBLAS
nvcc -O3 -arch=sm_86 src/cuda/matrix_cublas.cu -o matrix_cublas -lcublas

# Run
./matrix_cublas 2048
```

### Part 5: Image Convolution

```bash
# Compile convolution library
nvcc -O3 -std=c++11 -arch=sm_86 -shared -Xcompiler -fPIC \
     src/library/conv_lib.cu -o libgpukit.so

# Run Python convolution
python3 src/python/py_conv.py
```

**Supported Filters:**

| Filter | Kernel | Purpose |
|--------|--------|---------|
| Sobel X | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | Horizontal edge detection |
| Sobel Y | `[[-1,-2,-1],[0,0,0],[1,2,1]]` | Vertical edge detection |
| Laplacian | `[[0,-1,0],[-1,4,-1],[0,-1,0]]` | Edge detection |
| Gaussian 3×3 | `[[1,2,1],[2,4,2],[1,2,1]]` | Blur |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | Sharpening |
| Emboss | `[[-2,-1,0],[-1,1,1],[0,1,2]]` | Emboss effect |

### Part 6: Python Integration

```python
import ctypes
import numpy as np

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libgpukit.so")

# Configure function signature
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

# Create matrices
N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Call CUDA function
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
```

---

## 📊 Performance Analysis

### Scaling Behavior

#### CPU Performance (O(N³) Complexity)
- **512→1024**: 20× slower
- **1024→2048**: 8× slower
- Poor scaling due to sequential processing

#### GPU Performance (Better Parallelization)

**Naïve CUDA:**
- **512→1024**: 1.6× slower
- **1024→2048**: 6.4× slower

**Tiled CUDA:**
- **512→1024**: 1.4× slower
- **1024→2048**: 2.3× slower

**cuBLAS:**
- **512→1024**: 1.2× slower
- **1024→2048**: 1.1× slower

### Memory Transfer Overhead

For N=2048 matrices:
- **Data size**: 50.3 MB (3 × 2048² × 4 bytes)
- **PCIe bandwidth**: ~16 GB/s
- **Transfer time**: ~3.1 ms
- **Computation time**: 3.2 ms (cuBLAS)
- **Overhead ratio**: ~97%

**Key Insight**: For production applications, keep data resident on GPU to amortize transfer costs.

### Optimization Impact

| Optimization Technique | Improvement | Complexity |
|------------------------|-------------|------------|
| Naïve CUDA | 143-2,257× | Low |
| Shared Memory Tiling | 26-30% over naïve | Medium |
| cuBLAS | 2-18× over tiled | Low (API) |

---

## 🎓 Analysis & Insights

### 1. Performance Scaling with Matrix Size

**Observation**: GPU advantage increases dramatically with problem size.

| Size | CPU (ms) | GPU (ms) | Speedup |
|------|----------|----------|---------|
| 256×256 | 20.85 | 2.53 | 8.2× |
| 512×512 | 210.37 | 0.96 | 219.1× |
| 1024×1024 | 4,138.85 | 0.69 | 5,998.3× |

### 2. GPU Breakeven Point

- **Small matrices (< 32×32)**: CPU faster due to overhead
- **Medium matrices (32×32 - 512×512)**: GPU competitive
- **Large matrices (> 512×512)**: GPU dominates (100×+ speedup)

### 3. Tiling Optimization Benefits

**Memory Access Reduction:**
- Without tiling: N³ global memory accesses
- With tiling: N³/B shared memory accesses (B = tile size)
- Reduction factor: 16× for 16×16 tiles

**Performance Gain:**
- Small matrices (512): 1.01× improvement
- Medium matrices (1024): 1.13× improvement
- Large matrices (2048): 3.13× improvement

### 4. Why cuBLAS Outperforms Hand-Written Kernels

1. **Tensor Core Utilization**: Specialized hardware for matrix operations
2. **Advanced Memory Patterns**: Optimized bank conflict avoidance
3. **Multi-Kernel Fusion**: Reduces kernel launch overhead
4. **Auto-tuning**: Selects best algorithm for matrix size
5. **Assembly Optimizations**: Hand-tuned at instruction level
6. **Years of Engineering**: NVIDIA's expert optimization

### 5. Convolution Performance Insights

**Size Scaling:**
- Small images (256×256): 857× speedup
- Medium images (512×512): 1,872× speedup
- Large images (1024×1024): Similar speedup (overhead amortized)

**Kernel Size Impact:**
- Small kernels (3×3): Lower arithmetic intensity
- Large kernels (7×7): Better GPU utilization

---

## 🔍 Implementation Details

### Tiled Matrix Multiplication Kernel

```cuda
#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0;
    
    // Iterate over tiles
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tile into shared memory
        if (Row < N && (m*TILE_WIDTH+tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
            
        if (Col < N && (m*TILE_WIDTH+ty) < N)
            ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
            
        __syncthreads();
    }
    
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}
```

**Key Optimizations:**
- ✅ Shared memory reduces global memory accesses
- ✅ Coalesced memory access pattern
- ✅ Proper synchronization with `__syncthreads()`
- ✅ Boundary checking for non-multiple tile sizes

### 2D Convolution Kernel

```cuda
__global__ void conv2d_tiled(const uint32_t* img, int H, int W,
                              const int32_t* ker, int K,
                              int32_t* out) {
    __shared__ uint32_t tile[TILE_SIZE + MAX_K][TILE_SIZE + MAX_K];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Load tile with halo region
    // ... (boundary handling code)
    
    __syncthreads();
    
    // Compute convolution
    int32_t sum = 0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            sum += tile[ty + i][tx + j] * ker[i * K + j];
        }
    }
    
    if (row < H && col < W)
        out[row * W + col] = sum;
}
```

---

## 📈 Benchmarking & Results

### Complete Performance Table

| Implementation | N=512 | N=1024 | N=2048 | Units |
|----------------|-------|--------|--------|-------|
| **CPU (C)** | 211.02 | 4,246.33 | 52,961.16 | ms |
| **Naïve CUDA** | 0.13 | 0.92 | 7.04 | ms |
| **Tiled CUDA** | 0.10 | 0.71 | 5.77 | ms |
| **cuBLAS** | 0.03 | 0.12 | 0.76 | ms |

### GPU Metrics (NVIDIA RTX A6000)

| Metric | Value |
|--------|-------|
| Compute Capability | 8.6 |
| Global Memory | 47.41 GB |
| Shared Memory per Block | 48 KB |
| Max Threads per Block | 1024 |
| CUDA Cores | 10,752 |

---

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce matrix size or use batching
# Check available memory
nvidia-smi

# Run with smaller matrices
./matrix_cuda 512
```

**2. Kernel Launch Failure**
```bash
# Check CUDA error codes
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

**3. Slow Performance**
```bash
# Verify GPU is being used
nvidia-smi

# Check compiler optimizations
nvcc -O3 -arch=sm_86 ...  # Use appropriate compute capability
```

**4. Python Library Loading Issues**
```bash
# Ensure library is in current directory or LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./

# Check library dependencies
ldd libgpukit.so
```

---

## 📚 Resources

### Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Library Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Research Papers
- "Optimizing Matrix Multiply using Shared Memory" - NVIDIA
- "CUDA Performance Analysis and Optimization" - Harris, M.

### Tutorials
- [CUDA by Example](https://developer.nvidia.com/cuda-example)
- [GPU Gems Series](https://developer.nvidia.com/gpugems/gpugems/contributors)

---

## Acknowledgments

- NVIDIA for CUDA Toolkit and documentation
- Google Cloud Platform for GPU instances
- Course instructors and TAs for guidance
- Research community for optimization techniques

---
