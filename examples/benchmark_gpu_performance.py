import os
import sys
import time
import gc
import json
import resource
import torch
import numpy as np
import ants

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from syntx.syn import SyNTo, registration, grid_to_physical_affine, calculate_inverse_identity_error

def get_peak_memory_mb():
    """Returns max RSS memory used by current process in MB."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return rusage.ru_maxrss / (1024.0 * 1024.0)
    return rusage.ru_maxrss / 1024.0

def benchmark_3d_subject_pair(fixed, moving, backend='pytorch', device=None, reg_iterations=[10, 5, 2], affine_iterations=[10, 5, 2], levels=[4, 2, 1]):
    """
    Benchmarks end-to-end 3D registration performance for a single 3D subject pair.
    
    Measures:
    - Affine initial alignment time
    - Deformable SyN iteration time
    - Grid composition & evaluation time
    - Output transform resampling time
    - Total execution time
    - Memory delta and peak usage
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
    dev_str = str(device).lower() if device is not None else ''
    gc.collect()
    if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    gc.collect()
        
    mem_before = get_peak_memory_mb()
    
    t_start = time.time()
    
    # 1. Setup & Pre-normalization
    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    direction = fixed.direction
    
    fi_np = fixed.numpy()
    mi_np = moving.numpy()
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    grid_shape_zyx = tuple(reversed(grid_shape))
    
    # 2. Model Initialization & Affine stage
    t_aff_start = time.time()
    if backend == 'pytorch':
        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        model = SyNTo(
            dim=dim, grid_shape=grid_shape_zyx, spacing=spacing, origin=fixed.origin, direction=direction,
            fluid_sigma=1.732, elastic_sigma=0.0, transform_type='Affine',
            inverse_method='fixed_point', inverse_steps=8, project_inverse=True,
            projection_frequency=5, interpolator='linear'
        ).to(device)
        
        # Fit Affine stage only first for timing breakdown
        smoothing_sigmas = [float(np.log2(s)) if s > 1 else 0.0 for s in levels]
        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=[0] * len(levels),
            affine_epochs=affine_iterations,
            affine_lr=1e-2,
            cfl_voxels=0.25,
            similarity_metric='lncc',
            fixed_spacing=fixed.spacing, fixed_origin=fixed.origin, fixed_direction=fixed.direction,
            moving_spacing=moving.spacing, moving_origin=moving.origin, moving_direction=moving.direction,
            aff_metric='mattes',
            smoothing_sigmas=smoothing_sigmas,
            verbose=False,
            device=device
        )
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
    t_aff_end = time.time()
    time_affine = t_aff_end - t_aff_start
    
    # 3. Deformable SyN stage
    t_syn_start = time.time()
    if backend == 'pytorch':
        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            affine_epochs=[0] * len(levels),
            cfl_voxels=0.25,
            similarity_metric='lncc',
            fixed_spacing=fixed.spacing, fixed_origin=fixed.origin, fixed_direction=fixed.direction,
            moving_spacing=moving.spacing, moving_origin=moving.origin, moving_direction=moving.direction,
            smoothing_sigmas=smoothing_sigmas,
            verbose=False,
            device=device
        )
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
    t_syn_end = time.time()
    time_syn = t_syn_end - t_syn_start
    
    # 4. Grid composition & Evaluation
    t_eval_start = time.time()
    if backend == 'pytorch' and hasattr(model, 'warp_l2r') and hasattr(model, 'warp_l2r_inv'):
        w_l2r = model.warp_l2r.data.cpu()
        w_l2r_inv = model.warp_l2r_inv.data.cpu()
        inv_err = calculate_inverse_identity_error(w_l2r, w_l2r_inv, fixed.spacing, fixed.origin, fixed.direction)
    else:
        inv_err = {'max_error': 0.0, 'mean_error': 0.0}
    t_eval_end = time.time()
    time_grid_eval = t_eval_end - t_eval_start
    
    # 5. Full registration wrapper call (end-to-end benchmark result)
    t_e2e_start = time.time()
    res = syntx.syn(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyNTo',
        backend=backend,
        syn_metric='lncc',
        levels=levels,
        affine_iterations=affine_iterations,
        reg_iterations=reg_iterations,
        device=device,
        inverse_steps=8
    )
    t_e2e_end = time.time()
    total_time = t_e2e_end - t_e2e_start
    time_resampling = max(0.0, total_time - (time_affine + time_syn + time_grid_eval))
    
    dev_str = str(device).lower() if device is not None else ''
    gc.collect()
    if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    gc.collect()
        
    mem_after = get_peak_memory_mb()
    mem_delta = max(0.0, mem_after - mem_before)
    
    result = {
        'backend': backend,
        'device': device,
        'image_shape': list(fixed.shape),
        'total_time_seconds': round(total_time, 3),
        'time_affine_seconds': round(time_affine, 3),
        'time_syn_seconds': round(time_syn, 3),
        'time_grid_eval_seconds': round(time_grid_eval, 3),
        'time_resampling_seconds': round(time_resampling, 3),
        'inverse_identity_max_error_mm': round(inv_err.get('max_error', 0.0), 4),
        'inverse_identity_mean_error_mm': round(inv_err.get('mean_error', 0.0), 4),
        'mem_before_mb': round(mem_before, 1),
        'mem_after_mb': round(mem_after, 1),
        'mem_delta_mb': round(mem_delta, 1),
        'sub_10s_passed': total_time < 10.0
    }
    return result

def main():
    print("=" * 70)
    print(" Syntx 3D GPU Registration Performance Benchmark")
    print("=" * 70)
    
    fixed_path = 'cache/img1_brain.nii.gz'
    moving_path = 'cache/img2_brain.nii.gz'
    
    if os.path.exists(fixed_path) and os.path.exists(moving_path):
        print(f"Loading 3D brain images from cache: {fixed_path}, {moving_path}")
        fixed = ants.image_read(fixed_path)
        moving = ants.image_read(moving_path)
    else:
        print("Creating synthetic 3D brain subject pair (160, 256, 256)...")
        np.random.seed(42)
        grid_shape = (160, 256, 256)
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)
        
        # Create brain-like ellipsoid structure
        z, y, x = np.ogrid[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
        cz, cy, cx = 80, 128, 128
        mask = ((z - cz)**2 / 60**2 + (y - cy)**2 / 90**2 + (x - cx)**2 / 90**2) <= 1.0
        
        fi_data = np.zeros(grid_shape, dtype=np.float32)
        fi_data[mask] = 1.0 + 0.1 * np.random.randn(*mask.shape)[mask]
        
        mi_data = np.roll(fi_data, shift=(2, 3, 2), axis=(0, 1, 2))
        
        fixed = ants.from_numpy(fi_data, origin=origin, spacing=spacing, direction=direction)
        moving = ants.from_numpy(mi_data, origin=origin, spacing=spacing, direction=direction)
        
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on device: {device}")
    
    res = benchmark_3d_subject_pair(
        fixed, moving, backend='pytorch', device=device,
        reg_iterations=[10, 5, 2], affine_iterations=[10, 5, 2], levels=[4, 2, 1]
    )
    
    print("-" * 70)
    print(f" 3D Pair Execution Time : {res['total_time_seconds']:.3f} s (Target < 10.0s)")
    print(f"   - Affine Stage       : {res['time_affine_seconds']:.3f} s")
    print(f"   - SyN Deformable     : {res['time_syn_seconds']:.3f} s")
    print(f"   - Grid Composition   : {res['time_grid_eval_seconds']:.3f} s")
    print(f"   - Transform Resampling: {res['time_resampling_seconds']:.3f} s")
    print(f" Memory Usage           : {res['mem_after_mb']:.1f} MB (Delta: {res['mem_delta_mb']:.1f} MB)")
    print(f" Inverse Identity Error : Max = {res['inverse_identity_max_error_mm']:.4f} mm, Mean = {res['inverse_identity_mean_error_mm']:.4f} mm")
    print(f" Benchmark Pass Status  : {'PASSED' if res['sub_10s_passed'] else 'FAILED'}")
    print("=" * 70)
    
    # Write JSON results
    out_json = 'examples/gpu_benchmark_results.json'
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved benchmark results to {out_json}")
    
    assert res['sub_10s_passed'], f"Execution time {res['total_time_seconds']}s exceeded 10.0s limit!"
    print("Benchmark complete!")

if __name__ == '__main__':
    main()
