import os
import sys
import time
import gc
import torch
import numpy as np
import pytest
import ants

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from examples.benchmark_gpu_performance import benchmark_3d_subject_pair, get_peak_memory_mb

@pytest.fixture(scope="module")
def brain_3d_pair():
    """Provides a 3D brain subject pair for performance and memory benchmarking."""
    fixed_path = 'cache/img1_brain.nii.gz'
    moving_path = 'cache/img2_brain.nii.gz'
    
    if os.path.exists(fixed_path) and os.path.exists(moving_path):
        fixed = ants.image_read(fixed_path)
        moving = ants.image_read(moving_path)
    else:
        np.random.seed(42)
        grid_shape = (128, 128, 128)
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)
        
        z, y, x = np.ogrid[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
        cz, cy, cx = 64, 64, 64
        mask = ((z - cz)**2 / 40**2 + (y - cy)**2 / 50**2 + (x - cx)**2 / 50**2) <= 1.0
        
        fi_data = np.zeros(grid_shape, dtype=np.float32)
        fi_data[mask] = 1.0 + 0.1 * np.random.randn(*mask.shape)[mask]
        mi_data = np.roll(fi_data, shift=(2, 3, 2), axis=(0, 1, 2))
        
        fixed = ants.from_numpy(fi_data, origin=origin, spacing=spacing, direction=direction)
        moving = ants.from_numpy(mi_data, origin=origin, spacing=spacing, direction=direction)
        
    return fixed, moving

def test_gpu_3d_registration_sub_10s(brain_3d_pair):
    """Asserts end-to-end 3D registration executes in < 10.0 seconds per 3D subject pair."""
    fixed, moving = brain_3d_pair
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    t0 = time.time()
    res = syntx.syn(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyNTo',
        backend='pytorch',
        syn_metric='lncc',
        levels=[4, 2, 1],
        affine_iterations=[10, 5, 2],
        reg_iterations=[10, 5, 2],
        device=device,
        inverse_steps=8
    )
    t1 = time.time()
    total_time = t1 - t0
    
    # Verify performance threshold
    assert total_time < 10.0, f"3D SyN registration took {total_time:.2f}s, exceeding 10.0s benchmark limit!"
    
    # Verify outputs
    assert res['warpedmovout'] is not None
    assert 'fwdtransforms' in res
    assert len(res['fwdtransforms']) > 0
    assert res['warpedmovout'].shape == fixed.shape

def test_gpu_memory_leak_free(brain_3d_pair):
    """Profiles memory usage across multiple registrations and asserts zero memory leaks."""
    fixed, moving = brain_3d_pair
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    rss_samples = []
    for _ in range(3):
        dev_str = str(device).lower() if device is not None else ''
        gc.collect()
        if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()
            
        res = syntx.syn(
            fixed=fixed,
            moving=moving,
            type_of_transform='SyNTo',
            backend='pytorch',
            syn_metric='lncc',
            levels=[4, 2, 1],
            affine_iterations=[5, 3, 2],
            reg_iterations=[5, 3, 2],
            device=device,
            inverse_steps=6
        )
        dev_str = str(device).lower() if device is not None else ''
        gc.collect()
        if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()
            
        rss_samples.append(get_peak_memory_mb())
        
    # Unbounded memory growth assertion: diff between run 2 and run 3 must be tiny (< 150 MB)
    memory_growth = rss_samples[2] - rss_samples[1]
    assert memory_growth < 150.0, f"Detected potential memory leak: RSS grew by {memory_growth:.1f} MB between iterations!"

def test_gpu_timing_breakdown(brain_3d_pair):
    """Asserts detailed timing breakdown reporting across registration phases."""
    fixed, moving = brain_3d_pair
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    res = benchmark_3d_subject_pair(
        fixed, moving, backend='pytorch', device=device,
        reg_iterations=[10, 5, 2], affine_iterations=[10, 5, 2], levels=[4, 2, 1]
    )
    
    assert res['total_time_seconds'] < 10.0
    assert 'time_affine_seconds' in res
    assert 'time_syn_seconds' in res
    assert 'time_grid_eval_seconds' in res
    assert 'time_resampling_seconds' in res
    assert res['time_affine_seconds'] >= 0.0
    assert res['time_syn_seconds'] >= 0.0
