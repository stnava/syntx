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
    res1 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 10], initial_transform='identity', verbose=False)
    torch.manual_seed(42)
    res2 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 10], initial_transform='identity', verbose=False)
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
    res1 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 5], initial_transform='identity', device='cpu', verbose=False)
    torch.manual_seed(42)
    res2 = syntx.syn(fixed=f, moving=m, reg_iterations=[10, 5], initial_transform='identity', device='cpu', verbose=False)
    dt = time.time() - t0
    
    # 1. Check warped images match exactly
    w1 = res1['warpedmovout'].numpy()
    w2 = res2['warpedmovout'].numpy()
    diff = np.max(np.abs(w1 - w2))
    assert diff < 1e-4, f"3D Warped images differ by {diff:.6e}"
    
    # 2. Check execution speed (< 10.0s with coverage overhead)
    assert dt < 10.0, f"3D reproducibility test took too long: {dt:.2f}s"


def test_syngs_reproducibility():
    """Verify bitwise/exact reproducibility of SyNGS with bootstrap_mode='none' and optional 'antithetic'.

    affine_seed is pinned to a fixed constant (not tied to the geodesic-shooting `seed`)
    across every call here: robust_affine now always runs the initial alignment stage,
    and this test is specifically about the DEFORMABLE stage's seed-dependence, so the
    affine sub-stage must be held identical across calls to isolate that -- the old
    affine_iterations=0 achieved this by skipping the (now-removed) affine stage entirely;
    a fixed affine_seed is the closest equivalent now that alignment always runs.

    NOTE: bootstrap_mode='none' is passed explicitly in section 1 below (rather than relying
    on the function default) because syngs_registration()'s default bootstrap_mode was
    changed to 'antithetic' during the defaults-alignment session (per
    docs/provenance/best_parameters.json), which is intentionally seed-dependent/stochastic.
    The bootstrap_mode='none' determinism property itself is unchanged and still verified here.
    """
    f, m = create_synthetic_data_2d()

    # 1. Under bootstrap_mode='none', runs are bit-for-bit identical regardless of seed
    res1 = syntx.syngs(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], affine_seed=7, seed=42, device='cpu', verbose=False, bootstrap_mode='none')
    res2 = syntx.syngs(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], affine_seed=7, seed=99, device='cpu', verbose=False, bootstrap_mode='none')

    w1 = res1['warpedmovout'].numpy()
    w2 = res2['warpedmovout'].numpy()
    diff = np.max(np.abs(w1 - w2))
    assert diff == 0.0, f"SyNGS default warped images differ across runs: {diff:.6e}"

    # 2. Under bootstrap_mode='antithetic', identical seeds produce bit-for-bit identical results
    res_a1 = syntx.syngs(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], affine_seed=7,
                         bootstrap_mode='antithetic', seed=42, device='cpu', verbose=False)
    res_a2 = syntx.syngs(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], affine_seed=7,
                         bootstrap_mode='antithetic', seed=42, device='cpu', verbose=False)
    diff_a = np.max(np.abs(res_a1['warpedmovout'].numpy() - res_a2['warpedmovout'].numpy()))
    assert diff_a == 0.0, f"SyNGS antithetic warped images differ across identical seeds: {diff_a:.6e}"

    # 3. Under bootstrap_mode='antithetic', different seeds explore different paths
    res_a3 = syntx.syngs(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], affine_seed=7,
                         bootstrap_mode='antithetic', seed=99, device='cpu', verbose=False)
    seed_diff = np.max(np.abs(res_a1['warpedmovout'].numpy() - res_a3['warpedmovout'].numpy()))
    assert seed_diff > 0.0, "SyNGS antithetic different seeds should produce different exploration paths"


def test_syn_antithetic_reproducibility():
    """Verify bitwise/exact reproducibility of SyN with seeded Antithetic Bootstrapping."""
    f, m = create_synthetic_data_2d()

    # initial_transform='identity' skips the (now robust_affine-backed, default-on) affine
    # stage entirely -- this test is specifically about the deformable/SyN stage's antithetic
    # bootstrap seed-dependence, which robust_affine's now much more precise pre-alignment
    # was found to otherwise mask (it aligns these trivial synthetic circles so well the
    # residual SyN task converged identically regardless of bootstrap seed).
    res1 = syntx.syn(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], initial_transform='identity',
                     bootstrap_mode='antithetic', bootstrap_orig_weight=0.80, bootstrap_jitter_scale=0.10,
                     use_analytical_gradients=False, seed=42, device='cpu', verbose=False)
    res2 = syntx.syn(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], initial_transform='identity',
                     bootstrap_mode='antithetic', bootstrap_orig_weight=0.80, bootstrap_jitter_scale=0.10,
                     use_analytical_gradients=False, seed=42, device='cpu', verbose=False)

    w1 = res1['warpedmovout'].numpy()
    w2 = res2['warpedmovout'].numpy()
    diff = np.max(np.abs(w1 - w2))
    assert diff == 0.0, f"SyN antithetic warped images differ across identical seeds: {diff:.6e}"

    res3 = syntx.syn(fixed=f, moving=m, levels=[2, 1], reg_iterations=[5, 5], initial_transform='identity',
                     bootstrap_mode='antithetic', bootstrap_orig_weight=0.80, bootstrap_jitter_scale=0.10,
                     use_analytical_gradients=False, seed=99, device='cpu', verbose=False)
    w3 = res3['warpedmovout'].numpy()
    seed_diff = np.max(np.abs(w1 - w3))
    assert seed_diff > 0.0, "SyN antithetic different seeds should produce different exploration paths"
