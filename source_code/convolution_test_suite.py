#!/usr/bin/env python3
"""
Comprehensive Convolution Test Suite
Tests the CUDA convolution function with various image processing filters
"""

import ctypes
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json

# Load the CUDA convolution library
lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgpukit.so"))

# Configure function signature
lib.conv2d_cuda_u32.argtypes = [
    np.ctypeslib.ndpointer(np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(np.int32, ndim=1, flags="C_CONTIGUOUS"),
]

# Image processing filters
FILTERS = {
    # Edge Detection Filters
    'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32),
    'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32),
    'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int32),
    'laplacian_8': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int32),
    
    # Blur Filters
    'gaussian_3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32),
    'gaussian_5x5': np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], 
                              [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=np.int32),
    'box_3x3': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32),
    'box_5x5': np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 
                         [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int32),
    
    # Sharpening Filters
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int32),
    'unsharp_mask': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.int32),
    
    # Emboss Filter
    'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.int32),
}

def load_image_as_grayscale(image_path):
    """Load image and convert to grayscale uint32"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img.astype(np.uint32)

def apply_convolution_cuda(image, kernel):
    """Apply convolution using CUDA implementation"""
    H, W = image.shape
    K = kernel.shape[0]
    
    # Prepare output array
    output = np.zeros((H, W), dtype=np.int32)
    
    # Apply convolution
    lib.conv2d_cuda_u32(image.ravel(), H, W, kernel.ravel(), K, output.ravel())
    
    return output

def apply_convolution_cpu(image, kernel):
    """Apply convolution using CPU implementation for comparison"""
    H, W = image.shape
    K = kernel.shape[0]
    pad = K // 2
    
    # Pad the image
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros((H, W), dtype=np.int32)
    
    for y in range(H):
        for x in range(W):
            for ky in range(K):
                for kx in range(K):
                    output[y, x] += padded[y + ky, x + kx] * kernel[ky, kx]
    
    return output

def benchmark_convolution(image, kernel, iterations=10):
    """Benchmark convolution performance"""
    H, W = image.shape
    K = kernel.shape[0]
    
    # Warm up
    apply_convolution_cuda(image, kernel)
    
    # Benchmark CUDA
    cuda_times = []
    for _ in range(iterations):
        start = time.time()
        apply_convolution_cuda(image, kernel)
        cuda_times.append(time.time() - start)
    
    # Benchmark CPU (only for smaller images due to performance)
    cpu_times = []
    if H <= 512:  # Only benchmark CPU for smaller images
        for _ in range(iterations):
            start = time.time()
            apply_convolution_cpu(image, kernel)
            cpu_times.append(time.time() - start)
    
    return {
        'cuda_mean': np.mean(cuda_times),
        'cuda_std': np.std(cuda_times),
        'cpu_mean': np.mean(cpu_times) if cpu_times else None,
        'cpu_std': np.std(cpu_times) if cpu_times else None,
        'speedup': np.mean(cpu_times) / np.mean(cuda_times) if cpu_times else None
    }

def normalize_image(image):
    """Normalize image for display"""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max > img_min:
        return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return image.astype(np.uint8)

def create_visualization(original, results, filter_name, save_path):
    """Create visualization of convolution results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Convolution Results: {filter_name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # CUDA results
    for i, (kernel_name, result) in enumerate(results.items()):
        if i >= 5:  # Limit to 5 results
            break
        
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        normalized = normalize_image(result)
        axes[row, col].imshow(normalized, cmap='gray')
        axes[row, col].set_title(f'CUDA: {kernel_name}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test_different_sizes():
    """Test convolution with different image and kernel sizes"""
    print("Testing convolution with different sizes...")
    
    # Test configurations: (image_size, kernel_size)
    test_configs = [
        (256, 3),   # Small image, small kernel
        (512, 3),   # Medium image, small kernel
        (1024, 3),  # Large image, small kernel
        (512, 5),   # Medium image, medium kernel
        (1024, 5),  # Large image, medium kernel
        (1024, 7),  # Large image, large kernel
    ]
    
    results = []
    
    for img_size, kernel_size in test_configs:
        print(f"Testing {img_size}x{img_size} image with {kernel_size}x{kernel_size} kernel...")
        
        # Create test image
        test_image = np.random.randint(0, 256, (img_size, img_size), dtype=np.uint32)
        
        # Create test kernel
        test_kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)
        
        # Benchmark
        perf = benchmark_convolution(test_image, test_kernel)
        
        results.append({
            'image_size': img_size,
            'kernel_size': kernel_size,
            'performance': perf
        })
        
        print(f"  CUDA: {perf['cuda_mean']:.6f}s ± {perf['cuda_std']:.6f}s")
        if perf['cpu_mean']:
            print(f"  CPU:  {perf['cpu_mean']:.6f}s ± {perf['cpu_std']:.6f}s")
            print(f"  Speedup: {perf['speedup']:.2f}x")
    
    return results

def main():
    """Main test function"""
    print("CUDA Convolution Test Suite")
    print("=" * 50)
    
    # Load test images
    image_dir = "images"
    image_files = ["out-0.jpeg", "out-1.jpeg", "out-2.jpeg"]
    
    all_results = {}
    performance_data = []
    
    # Test each image
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\nProcessing {img_file}...")
        
        # Load image
        original = load_image_as_grayscale(img_path)
        print(f"  Image size: {original.shape}")
        
        # Test with different filters
        filter_results = {}
        for filter_name, kernel in FILTERS.items():
            print(f"  Testing {filter_name} filter...")
            
            # Apply convolution
            result = apply_convolution_cuda(original, kernel)
            filter_results[filter_name] = result
            
            # Benchmark performance
            perf = benchmark_convolution(original, kernel)
            performance_data.append({
                'image': img_file,
                'filter': filter_name,
                'image_size': original.shape[0],
                'kernel_size': kernel.shape[0],
                'performance': perf
            })
            
            print(f"    CUDA time: {perf['cuda_mean']:.6f}s")
            if perf['speedup']:
                print(f"    Speedup: {perf['speedup']:.2f}x")
        
        all_results[img_file] = filter_results
        
        # Create visualization for first image with edge detection filters
        if img_file == image_files[0]:
            edge_filters = {k: v for k, v in filter_results.items() 
                          if k.startswith(('sobel', 'laplacian'))}
            create_visualization(original, edge_filters, 'Edge Detection', 
                               f'convolution_results_{img_file.replace(".jpeg", "")}.png')
    
    # Test different sizes
    print("\n" + "=" * 50)
    size_results = test_different_sizes()
    
    # Save performance data
    with open('convolution_performance.json', 'w') as f:
        json.dump({
            'image_results': performance_data,
            'size_results': size_results
        }, f, indent=2)
    
    # Generate summary report
    generate_summary_report(performance_data, size_results)
    
    print("\nTest completed! Results saved to:")
    print("  - convolution_performance.json")
    print("  - convolution_summary_report.txt")
    print("  - convolution_results_*.png")

def generate_summary_report(performance_data, size_results):
    """Generate a summary report of convolution performance"""
    with open('convolution_summary_report.txt', 'w') as f:
        f.write("CUDA Convolution Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Image processing results
        f.write("Image Processing Performance:\n")
        f.write("-" * 30 + "\n")
        for result in performance_data:
            perf = result['performance']
            f.write(f"Image: {result['image']}, Filter: {result['filter']}\n")
            f.write(f"  Size: {result['image_size']}x{result['image_size']}, Kernel: {result['kernel_size']}x{result['kernel_size']}\n")
            f.write(f"  CUDA time: {perf['cuda_mean']:.6f}s ± {perf['cuda_std']:.6f}s\n")
            if perf['speedup']:
                f.write(f"  CPU time: {perf['cpu_mean']:.6f}s ± {perf['cpu_std']:.6f}s\n")
                f.write(f"  Speedup: {perf['speedup']:.2f}x\n")
            f.write("\n")
        
        # Size scaling results
        f.write("\nSize Scaling Performance:\n")
        f.write("-" * 30 + "\n")
        for result in size_results:
            perf = result['performance']
            f.write(f"Image: {result['image_size']}x{result['image_size']}, Kernel: {result['kernel_size']}x{result['kernel_size']}\n")
            f.write(f"  CUDA time: {perf['cuda_mean']:.6f}s ± {perf['cuda_std']:.6f}s\n")
            if perf['speedup']:
                f.write(f"  CPU time: {perf['cpu_mean']:.6f}s ± {perf['cpu_std']:.6f}s\n")
                f.write(f"  Speedup: {perf['speedup']:.2f}x\n")
            f.write("\n")

if __name__ == "__main__":
    main()
