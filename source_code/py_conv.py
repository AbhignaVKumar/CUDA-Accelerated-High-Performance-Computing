import ctypes, numpy as np, time, os

lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgpukit.so"))

# matmul
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.gpu_matrix_multiply.restype = None

# conv
lib.conv2d_cuda_u32.argtypes = [
    np.ctypeslib.ndpointer(np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(np.int32,  ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(np.int32,  ndim=1, flags="C_CONTIGUOUS"),
]
lib.conv2d_cuda_u32.restype = None

# quick test
H, W = 1024, 1024
img = (np.random.rand(H, W)*255).astype(np.uint32)
ker = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.int32)
out = np.zeros((H, W), dtype=np.int32)

t0 = time.time()
lib.conv2d_cuda_u32(img.ravel(), H, W, ker.ravel(), ker.shape[0], out.ravel())
t1 = time.time()
print(f"conv_cuda HxW={H}x{W} K={ker.shape[0]} wall={t1-t0:.6f}s")

