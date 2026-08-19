import os
import time
import numpy as np
import torch
import ants
import pytest
import syntx

def create_synthetic_data_2d(shape=(32, 32)):
    x, y = np.ogrid[:shape[0], :shape[1]]
    center1 = (16, 16)
    center2 = (14, 18)
    
    img1 = ((x - center1[0])**2 + (y - center1[1])**2 <= 8**2).astype(np.float32)
    img2 = ((x - center2[0])**2 + (y - center2[1])**2 <= 8**2).astype(np.float32)
    
    f = ants.from_numpy(img1, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    m = ants.from_numpy(img2, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    return f, m

def create_synthetic_data_3d(shape=(24, 24, 24)):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center1 = (12, 12, 12)
    center2 = (10, 14, 11)
    
    img1 = ((z - center1[0])**2 + (y - center1[1])**2 + (x - center1[2])**2 <= 6**2).astype(np.float32)
    img2 = ((z - center2[0])**2 + (y - center2[1])**2 + (x - center2[2])**2 <= 6**2).astype(np.float32)
    
    f = ants.from_numpy(img1, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    m = ants.from_numpy(img2, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    return f, m

def test_fast_reproducibility_2d():
    """Verify exact bitwise/float reproducibility of 2D SyN registration across consecutive runs."""
    f, m = create_synthetic_data_2d()
    
    t0 = time.time()
    torch.manual_seed(42)
    res1 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 10], affine_iterations=[10, 10], verbose=False)
    torch.manual_seed(42)
    res2 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 10], affine_iterations=[10, 10], verbose=False)
    dt = time.time() - t0
    
    # 1. Check warped images match exactly
    w1 = res1['warpedmovout'].numpy()
    w2 = res2['warpedmovout'].numpy()
    diff = np.max(np.abs(w1 - w2))
    assert diff < 1e-4, f"2D Warped images differ by {diff:.6e}"
    
    # 2. Check execution speed (< 10.0s with coverage overhead)
    assert dt < 10.0, f"2D reproducibility test took too long: {dt:.2f}s"

def test_fast_reproducibility_3d():
    """Verify exact bitwise/float reproducibility of 3D SyN registration across consecutive runs."""
    f, m = create_synthetic_data_3d()
    
    t0 = time.time()
    torch.manual_seed(42)
    res1 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 5], affine_iterations=[10, 5], verbose=False)
    torch.manual_seed(42)
    res2 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 5], affine_iterations=[10, 5], verbose=False)
    dt = time.time() - t0
    
    # 1. Check warped images match exactly
    w1 = res1['warpedmovout'].numpy()
    w2 = res2['warpedmovout'].numpy()
    diff = np.max(np.abs(w1 - w2))
    assert diff < 1e-4, f"3D Warped images differ by {diff:.6e}"
    
    # 2. Check execution speed (< 10.0s with coverage overhead)
    assert dt < 10.0, f"3D reproducibility test took too long: {dt:.2f}s"
