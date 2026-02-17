#!/usr/bin/env python3
"""
Accelerated Python Program using CUDA-Enhanced Library
Demonstrates various CUDA-accelerated operations and compares performance
"""

import ctypes
import numpy as np
import time
import os
import sys
import json
from typing import Dict, List, Tuple

# Load the enhanced CUDA library
lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libenhanced.so"))

# Configure function signatures
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]

lib.gpu_vector_add.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]

lib.gpu_matrix_transpose.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
]

lib.gpu_matrix_elementwise.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
]

lib.gpu_vector_sum.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.gpu_vector_sum.restype = ctypes.c_float

lib.conv2d_cuda_u32.argtypes = [
    np.ctypeslib.ndpointer(np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(np.int32, ndim=1, flags="C_CONTIGUOUS"),
]

class CUDAAcceleratedProcessor:
    """Main class for CUDA-accelerated operations"""
    
    def __init__(self):
        self.results = {}
        
    def matrix_multiplication(self, size: int, iterations: int = 10) -> Dict:
        """Perform matrix multiplication with CUDA acceleration"""
        print(f"Testing Matrix Multiplication ({size}x{size})...")
        
        # Generate test data
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        C = np.zeros((size, size), dtype=np.float32)
        
        # Warm up
        lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), size)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), size)
            times.append(time.time() - start)
        
        # Verify result
        expected = A @ B
        max_error = np.max(np.abs(expected - C))
        
        return {
            'operation': 'matrix_multiplication',
            'size': size,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'max_error': max_error,
            'gflops': (2 * size**3) / (np.mean(times) * 1e9)
        }
    
    def vector_addition(self, size: int, iterations: int = 100) -> Dict:
        """Perform vector addition with CUDA acceleration"""
        print(f"Testing Vector Addition (size: {size})...")
        
        # Generate test data
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        c = np.zeros(size, dtype=np.float32)
        
        # Warm up
        lib.gpu_vector_add(a, b, c, size)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            lib.gpu_vector_add(a, b, c, size)
            times.append(time.time() - start)
        
        # Verify result
        expected = a + b
        max_error = np.max(np.abs(expected - c))
        
        return {
            'operation': 'vector_addition',
            'size': size,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'max_error': max_error
        }
    
    def matrix_transpose(self, rows: int, cols: int, iterations: int = 50) -> Dict:
        """Perform matrix transpose with CUDA acceleration"""
        print(f"Testing Matrix Transpose ({rows}x{cols})...")
        
        # Generate test data
        input_matrix = np.random.rand(rows, cols).astype(np.float32)
        output_matrix = np.zeros((cols, rows), dtype=np.float32)
        
        # Warm up
        lib.gpu_matrix_transpose(input_matrix.ravel(), output_matrix.ravel(), rows, cols)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            lib.gpu_matrix_transpose(input_matrix.ravel(), output_matrix.ravel(), rows, cols)
            times.append(time.time() - start)
        
        # Verify result
        expected = input_matrix.T
        max_error = np.max(np.abs(expected.ravel() - output_matrix.ravel()))
        
        return {
            'operation': 'matrix_transpose',
            'rows': rows,
            'cols': cols,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'max_error': max_error
        }
    
    def elementwise_operations(self, size: int, iterations: int = 50) -> Dict:
        """Perform element-wise operations with CUDA acceleration"""
        print(f"Testing Element-wise Operations (size: {size})...")
        
        # Generate test data
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32) + 0.1  # Avoid division by zero
        c = np.zeros(size, dtype=np.float32)
        
        operations = ['addition', 'multiplication', 'division', 'power']
        results = {}
        
        for op_id, op_name in enumerate(operations):
            # Warm up
            lib.gpu_matrix_elementwise(a, b, c, size, op_id)
            
            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.time()
                lib.gpu_matrix_elementwise(a, b, c, size, op_id)
                times.append(time.time() - start)
            
            # Verify result
            if op_id == 0:  # Addition
                expected = a + b
            elif op_id == 1:  # Multiplication
                expected = a * b
            elif op_id == 2:  # Division
                expected = np.where(b != 0, a / b, 0)
            elif op_id == 3:  # Power
                expected = np.power(a, b)
            
            max_error = np.max(np.abs(expected - c))
            
            results[op_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'max_error': max_error
            }
        
        return {
            'operation': 'elementwise_operations',
            'size': size,
            'operations': results
        }
    
    def vector_reduction(self, size: int, iterations: int = 50) -> Dict:
        """Perform vector sum reduction with CUDA acceleration"""
        print(f"Testing Vector Sum Reduction (size: {size})...")
        
        # Generate test data
        input_vector = np.random.rand(size).astype(np.float32)
        
        # Warm up
        lib.gpu_vector_sum(input_vector, size)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            result = lib.gpu_vector_sum(input_vector, size)
            times.append(time.time() - start)
        
        # Verify result
        expected = np.sum(input_vector)
        error = abs(expected - result)
        
        return {
            'operation': 'vector_reduction',
            'size': size,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'error': error,
            'result': result
        }
    
    def convolution_processing(self, image_size: int, iterations: int = 10) -> Dict:
        """Perform convolution processing with CUDA acceleration"""
        print(f"Testing Convolution Processing ({image_size}x{image_size})...")
        
        # Generate test image and kernel
        image = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint32)
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)  # Sobel X
        output = np.zeros((image_size, image_size), dtype=np.int32)
        
        # Warm up
        lib.conv2d_cuda_u32(image.ravel(), image_size, image_size, 
                           kernel.ravel(), 3, output.ravel())
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            lib.conv2d_cuda_u32(image.ravel(), image_size, image_size, 
                               kernel.ravel(), 3, output.ravel())
            times.append(time.time() - start)
        
        return {
            'operation': 'convolution_processing',
            'image_size': image_size,
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark of all operations"""
        print("CUDA-Accelerated Python Program - Comprehensive Benchmark")
        print("=" * 60)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'operations': []
        }
        
        # Test different sizes
        test_sizes = [256, 512, 1024]
        
        for size in test_sizes:
            # Matrix multiplication
            results['operations'].append(self.matrix_multiplication(size))
            
            # Vector addition
            results['operations'].append(self.vector_addition(size * size))
            
            # Matrix transpose
            results['operations'].append(self.matrix_transpose(size, size))
            
            # Element-wise operations
            results['operations'].append(self.elementwise_operations(size * size))
            
            # Vector reduction
            results['operations'].append(self.vector_reduction(size * size))
            
            # Convolution
            results['operations'].append(self.convolution_processing(size))
        
        return results
    
    def save_results(self, results: Dict, filename: str = 'cuda_python_results.json'):
        """Save benchmark results to JSON file"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        converted_results = convert_numpy_types(results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print summary of benchmark results"""
        print("\n" + "=" * 60)
        print("CUDA-ACCELERATED PYTHON PROGRAM - PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for op in results['operations']:
            if op['operation'] == 'matrix_multiplication':
                print(f"Matrix Multiplication ({op['size']}x{op['size']}):")
                print(f"  Time: {op['mean_time']:.6f}s ± {op['std_time']:.6f}s")
                print(f"  GFLOPS: {op['gflops']:.2f}")
                print(f"  Max Error: {op['max_error']:.2e}")
            elif op['operation'] == 'vector_addition':
                print(f"Vector Addition (size: {op['size']}):")
                print(f"  Time: {op['mean_time']:.6f}s ± {op['std_time']:.6f}s")
                print(f"  Max Error: {op['max_error']:.2e}")
            elif op['operation'] == 'matrix_transpose':
                print(f"Matrix Transpose ({op['rows']}x{op['cols']}):")
                print(f"  Time: {op['mean_time']:.6f}s ± {op['std_time']:.6f}s")
                print(f"  Max Error: {op['max_error']:.2e}")
            elif op['operation'] == 'elementwise_operations':
                print(f"Element-wise Operations (size: {op['size']}):")
                for op_name, op_result in op['operations'].items():
                    print(f"  {op_name.capitalize()}: {op_result['mean_time']:.6f}s ± {op_result['std_time']:.6f}s")
            elif op['operation'] == 'vector_reduction':
                print(f"Vector Sum Reduction (size: {op['size']}):")
                print(f"  Time: {op['mean_time']:.6f}s ± {op['std_time']:.6f}s")
                print(f"  Error: {op['error']:.2e}")
            elif op['operation'] == 'convolution_processing':
                print(f"Convolution Processing ({op['image_size']}x{op['image_size']}):")
                print(f"  Time: {op['mean_time']:.6f}s ± {op['std_time']:.6f}s")
            print()

def main():
    """Main function"""
    processor = CUDAAcceleratedProcessor()
    
    # Run comprehensive benchmark
    results = processor.run_comprehensive_benchmark()
    
    # Save results
    processor.save_results(results)
    
    # Print summary
    processor.print_summary(results)
    
    print("CUDA-accelerated Python program completed successfully!")

if __name__ == "__main__":
    main()
