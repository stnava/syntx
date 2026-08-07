import time
import torch
import math
import syntx

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# Benchmark 2D and 3D Sobolev filtering with and without caching
shape_2d = (256, 256)
x_2d = torch.randn(3, 256, 256, 2, device=device)

# Un-cached 2D test
t0 = time.time()
for _ in range(100):
    k_axes = []
    for d in range(2):
        n_d = shape_2d[d]
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
        k_axes.append(lambda_d)
    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    lambda_sq = sum(k_j for k_j in k_mesh)
    K_dst = 1.0 / ((1.0 + 0.5 * lambda_sq) ** 2.0)
t1 = time.time()
print(f"2D Un-cached Kernel Construction (100x): {t1 - t0:.4f} seconds")

# Cached 2D test
cache_key = (shape_2d, 0.5, 2.0)
kernel_cache = {cache_key: K_dst}
t0 = time.time()
for _ in range(100):
    K_cached = kernel_cache[cache_key]
t1 = time.time()
print(f"2D Cached Kernel Lookup (100x): {t1 - t0:.6f} seconds")
