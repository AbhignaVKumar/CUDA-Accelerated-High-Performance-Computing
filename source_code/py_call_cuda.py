import ctypes, numpy as np, time, os

lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libmatrix.so"))
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.gpu_matrix_multiply.restype = None

N = int(os.environ.get("N", "1024"))
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

t0 = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
t1 = time.time()

# quick check vs NumPy (CPU) for small N; skip for huge N
if N <= 1024:
    ref = (A @ B)
    max_err = np.max(np.abs(ref - C))
    print(f"max_abs_err={max_err:.3e}")

print(f"N={N} tiled_cuda_via_ctypes wall={t1 - t0:.6f}s")

