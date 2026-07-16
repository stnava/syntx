import time
import torch
import numpy as np

# A simplified physical to normalized function with pre-allocated tensors
def physical_to_normalized_cached(phys_coords, target_shape, spacing_t, origin_t, direction_t, shape_t):
    dim = len(target_shape)
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    norm_coords_reversed = torch.flip(norm_coords, dims=[-1])
    return norm_coords_reversed.view(phys_coords.shape)

def profile_mitigation():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Profiling on device: {device}")
    
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    
    # Pre-allocate/pre-compute inputs
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = direction[::-1, ::-1].copy()
    
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=torch.float32)
    origin_t = torch.tensor(origin_rev, device=device, dtype=torch.float32)
    direction_t = torch.tensor(direction_rev, device=device, dtype=torch.float32)
    shape_t = torch.tensor(list(shape), device=device, dtype=torch.float32)
    
    from syntx.syn import get_physical_grid_torch
    # Pre-compute X_phys
    X_phys = get_physical_grid_torch(shape, spacing, origin, direction, device=device)
    
    n_iters = 100
    
    # Warm up
    phi = X_phys + torch.zeros_like(X_phys)
    norm = physical_to_normalized_cached(phi, shape, spacing_t, origin_t, direction_t, shape_t)
    
    # 1. Profile cached physical grid application + normalization
    t0 = time.perf_counter()
    for _ in range(n_iters):
        phi = X_phys + torch.zeros_like(X_phys)
        norm = physical_to_normalized_cached(phi, shape, spacing_t, origin_t, direction_t, shape_t)
        if device.type == 'mps':
            torch.mps.synchronize()
    t_conv_cached = time.perf_counter() - t0
    print(f"Time for {n_iters} cached physical space conversions: {t_conv_cached:.4f}s ({t_conv_cached/n_iters:.6f}s per iteration)")
    
    # 2. Profile a full registration epoch with caching mitigation
    fixed = torch.randn((1, 1) + shape, device=device)
    moving = torch.randn((1, 1) + shape, device=device)
    warp_l2r = torch.zeros((1,) + shape + (3,), device=device, requires_grad=True)
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        phi = X_phys + warp_l2r
        norm_coords = physical_to_normalized_cached(phi, shape, spacing_t, origin_t, direction_t, shape_t)
        warped = torch.nn.functional.grid_sample(moving, norm_coords, align_corners=True)
        loss = torch.mean((warped - fixed) ** 2)
        loss.backward()
        warp_l2r.grad.zero_()
        if device.type == 'mps':
            torch.mps.synchronize()
            
    t_total_cached = time.perf_counter() - t0
    print(f"Time for {n_iters} registration epochs (simulated, cached): {t_total_cached:.4f}s ({t_total_cached/n_iters:.6f}s per epoch)")
    print(f"Physical space conversion overhead (cached): {t_conv_cached / t_total_cached * 100:.2f}%")
    
    # Speedup calculation
    # (using values from previous run: t_total = 0.6987s, t_conv = 0.5030s)
    t_conv_prev = 0.5030
    t_total_prev = 0.6987
    speedup = (t_total_prev / t_total_cached - 1.0) * 100
    print(f"Expected speedup in registration epoch runtime: {speedup:.2f}%")

if __name__ == '__main__':
    profile_mitigation()
