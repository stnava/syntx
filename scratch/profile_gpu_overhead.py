import time
import torch
import numpy as np
import ants
from syntx.syn import SyNTo, get_physical_grid_torch, physical_to_normalized_torch

def profile_overhead():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Profiling on device: {device}")
    
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    
    # Let's warm up
    grid = get_physical_grid_torch(shape, spacing, origin, direction, device=device)
    norm = physical_to_normalized_torch(grid, shape, spacing, origin, direction)
    
    # 1. Profile grid creation + normalization
    n_iters = 100
    t0 = time.perf_counter()
    for _ in range(n_iters):
        grid = get_physical_grid_torch(shape, spacing, origin, direction, device=device)
        norm = physical_to_normalized_torch(grid, shape, spacing, origin, direction)
        if device.type == 'mps':
            torch.mps.synchronize()
    t_conv = time.perf_counter() - t0
    print(f"Time for {n_iters} physical space conversions: {t_conv:.4f}s ({t_conv/n_iters:.6f}s per iteration)")
    
    # 2. Profile a full registration epoch (simulated)
    # We will measure the forward pass, loss, backward pass, and update.
    # Fixed and moving images
    fixed = torch.randn((1, 1) + shape, device=device)
    moving = torch.randn((1, 1) + shape, device=device)
    warp_l2r = torch.zeros((1,) + shape + (3,), device=device, requires_grad=True)
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        # Physical space grid creation (what's done in prepare_mid_images_and_gradients_torch)
        grid = get_physical_grid_torch(shape, spacing, origin, direction, device=device)
        phi = grid + warp_l2r
        norm_coords = physical_to_normalized_torch(phi, shape, spacing, origin, direction)
        
        # Warp image
        warped = torch.nn.functional.grid_sample(moving, norm_coords, align_corners=True)
        
        # Loss
        loss = torch.mean((warped - fixed) ** 2)
        
        # Backward
        loss.backward()
        
        # Zero grad
        warp_l2r.grad.zero_()
        
        if device.type == 'mps':
            torch.mps.synchronize()
            
    t_total = time.perf_counter() - t0
    print(f"Time for {n_iters} registration epochs (simulated): {t_total:.4f}s ({t_total/n_iters:.6f}s per epoch)")
    print(f"Physical space conversion overhead: {t_conv / t_total * 100:.2f}%")

if __name__ == '__main__':
    profile_overhead()
