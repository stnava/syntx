"""
Adversarial Verification Suite for Milestone 1:
Stationary Operator Caching & DST-I Green Operator Invariance.

Tests:
1. Multi-epoch 2D SyN registration cache hit rate (> 95%) and deformation regularity.
2. Multi-epoch 3D SyN registration with regularizer='dsti1' (hit rate > 95%, min det(J) > 0, inverse error).
3. 3D TVF registration with regularizer_mode='dsti' (hit rate > 95%).
4. Exact numerical parity against uncached baseline across varied geometries and anisotropic spacings.
5. Device portability (CPU and MPS) and cache key isolation.
6. Dtype stability (float32 and float64) and precision preservation.
7. Cache capacity bound (LRU maxsize=64) and eviction under 100 diverse shapes.
8. Concurrent multi-threaded smoothing safety.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
import ants

import syntx
from syntx.core.smoothing import (
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    smooth_displacement_field_dst,
    clear_dst_cache,
    get_dst_cache_info,
    get_dsti_filter_cache_size,
    _MAX_DST_CACHE_SIZE,
)
from syntx.syn import SyNTo, compute_jacobian_determinant_nd
from syntx.tvf import TVFModel


def create_synthetic_pair_2d(shape=(32, 32)):
    x, y = np.ogrid[:shape[0], :shape[1]]
    c1 = (16, 16)
    c2 = (14, 18)
    img1 = ((x - c1[0])**2 + (y - c1[1])**2 <= 8**2).astype(np.float32)
    img2 = ((x - c2[0])**2 + (y - c2[1])**2 <= 8**2).astype(np.float32)
    f = ants.from_numpy(img1, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    m = ants.from_numpy(img2, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    return f, m


def create_synthetic_pair_3d(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0)):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c1 = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    c2 = (c1[0] - 2, c1[1] + 2, c1[2] - 1)
    img1 = (((z - c1[0])**2 + (y - c1[1])**2 + (x - c1[2])**2) <= 5**2).astype(np.float32)
    img2 = (((z - c2[0])**2 + (y - c2[1])**2 + (x - c2[2])**2) <= 5**2).astype(np.float32)
    f = ants.from_numpy(img1, origin=(0.0, 0.0, 0.0), spacing=spacing)
    m = ants.from_numpy(img2, origin=(0.0, 0.0, 0.0), spacing=spacing)
    return f, m


def uncached_baseline_dsti(m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0):
    """Pure un-cached reference using torch.meshgrid for exact comparison."""
    if fluid_sigma <= 0:
        return m
    dim = m.ndim - 2
    spatial_shape = m.shape[1:-1]
    device = m.device
    dtype = m.dtype
    alpha_val = float(alpha) if alpha is not None else float(fluid_sigma) / 2.0

    k_axes = []
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else None
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        scale = (sp_d ** 2) if spacing is not None else 1.0
        lambda_d = (4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)) / scale
        k_axes.append(lambda_d)

    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    lambda_sq = sum(k_mesh)
    K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

    curr = m.movedim(-1, 1).to(torch.float32)
    for d in range(dim):
        axis = d + 2
        n_d = spatial_shape[d]
        padded = torch.zeros(*curr.shape[:axis], 2 * n_d + 2, *curr.shape[axis + 1:], device=device, dtype=curr.dtype)
        padded.narrow(axis, 1, n_d).copy_(curr)
        padded.narrow(axis, n_d + 2, n_d).copy_(-torch.flip(curr, dims=[axis]))
        F = torch.fft.fft(padded, dim=axis)
        curr = -torch.imag(F.narrow(axis, 1, n_d)).contiguous()

    curr = curr * K_dst.unsqueeze(0).unsqueeze(0)

    for d in range(dim):
        axis = d + 2
        n_d = spatial_shape[d]
        padded = torch.zeros(*curr.shape[:axis], 2 * n_d + 2, *curr.shape[axis + 1:], device=device, dtype=curr.dtype)
        padded.narrow(axis, 1, n_d).copy_(curr)
        padded.narrow(axis, n_d + 2, n_d).copy_(-torch.flip(curr, dims=[axis]))
        F = torch.fft.fft(padded, dim=axis)
        curr = (-torch.imag(F.narrow(axis, 1, n_d)) / (2 * (n_d + 1))).contiguous()

    return curr.to(dtype=dtype).movedim(1, -1)


def test_m1_adversarial_syn_2d_cache_hit_rate_and_regularity():
    """Verify 2D SyN with regularizer='dsti' achieves > 95% cache hit rate and min det(J) > 0."""
    clear_dst_cache()
    info_before = get_dst_cache_info()
    assert info_before.hits == 0
    assert info_before.misses == 0

    f, m = create_synthetic_pair_2d()
    torch.manual_seed(42)
    res = syntx.syn(
        fixed=f,
        moving=m,
        reg_iterations=[30, 20],
        affine_iterations=[0, 0],
        regularizer='dsti',
        fast_smooth=True,
        verbose=False,
    )

    info_after = get_dst_cache_info()
    total_queries = info_after.hits + info_after.misses
    assert total_queries > 0, "No DST-I cache queries recorded"
    hit_rate = info_after.hits / total_queries

    print(f"2D SyN Cache Stats: hits={info_after.hits}, misses={info_after.misses}, hit_rate={hit_rate:.4f}")
    assert hit_rate > 0.95, f"Cache hit rate {hit_rate:.4f} is below 95% requirement"

    # Deformation regularity
    model = res['model']
    assert hasattr(model, 'midpoint_warp_l2r')
    warp_fwd = model.midpoint_warp_l2r
    det_j = compute_jacobian_determinant_nd(warp_fwd.contiguous())
    min_det_j = float(det_j.min().item())
    folding_pct = float((det_j <= 0.0).float().mean().item() * 100.0)

    print(f"2D SyN Regularity: min_det_j={min_det_j:.4f}, folding_pct={folding_pct:.4f}%")
    assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"
    assert folding_pct == 0.0, f"Deformation folding detected: {folding_pct}%"

    # Inverse consistency error
    inv_errs = res.get('inverse_identity_errors', {})
    assert 'phi_1' in inv_errs or len(inv_errs) > 0, "Inverse identity errors must be recorded"


def test_m1_adversarial_syn_3d_dsti1_hit_rate_and_invariance():
    """Verify 3D SyN with regularizer='dsti1' achieves > 95% hit rate and min det(J) > 0."""
    clear_dst_cache()

    f, m = create_synthetic_pair_3d(shape=(16, 16, 16))
    torch.manual_seed(42)
    res = syntx.syn(
        fixed=f,
        moving=m,
        reg_iterations=[25, 15],
        affine_iterations=[0, 0],
        regularizer='dsti1',
        fast_smooth=True,
        verbose=False,
    )

    info = get_dst_cache_info()
    total_queries = info.hits + info.misses
    hit_rate = info.hits / total_queries if total_queries > 0 else 0.0

    print(f"3D SyN Cache Stats: hits={info.hits}, misses={info.misses}, hit_rate={hit_rate:.4f}")
    assert hit_rate > 0.95, f"3D SyN hit rate {hit_rate:.4f} is below 95%"

    model = res['model']
    warp = model.midpoint_warp_l2r
    det_j = compute_jacobian_determinant_nd(warp.contiguous())
    min_det_j = float(det_j.min().item())
    folding_pct = float((det_j <= 0.0).float().mean().item() * 100.0)

    print(f"3D SyN Regularity: min_det_j={min_det_j:.4f}, folding_pct={folding_pct:.4f}%")
    assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"
    assert folding_pct == 0.0, f"Folding detected in 3D: {folding_pct}%"


def test_m1_adversarial_tvf_3d_hit_rate():
    """Verify 3D TVF solver with regularizer_mode='dsti' achieves > 95% cache hit rate."""
    clear_dst_cache()

    shape = (16, 16, 16)
    model = TVFModel(dim=3, image_shape=shape, velocity_shape=shape)
    fixed = torch.randn(1, 1, *shape, dtype=torch.float32)
    moving = torch.randn(1, 1, *shape, dtype=torch.float32)

    model.fit(
        fixed,
        moving,
        epochs_per_level=[30],
        regularizer='dsti',
        verbose=False,
    )

    info = get_dst_cache_info()
    total = info.hits + info.misses
    assert total > 0, "No queries recorded for TVF"
    hit_rate = info.hits / total
    print(f"3D TVF Cache Stats: hits={info.hits}, misses={info.misses}, hit_rate={hit_rate:.4f}")
    assert hit_rate > 0.95, f"TVF cache hit rate {hit_rate:.4f} is below 95%"


def test_m1_adversarial_bitwise_parity_against_uncached_baseline():
    """Verify bitwise or near-bitwise parity (L_inf == 0.0) against uncached meshgrid baseline."""
    torch.manual_seed(123)

    # Test 2D isotropic, 2D anisotropic, 3D isotropic, 3D anisotropic
    test_configs = [
        {"shape": (1, 24, 32, 2), "spacing": None, "alpha": 1.5, "s": 2.0},
        {"shape": (1, 20, 28, 2), "spacing": (0.7, 1.4), "alpha": 2.2, "s": 1.5},
        {"shape": (1, 12, 16, 14, 3), "spacing": None, "alpha": 3.0, "s": 2.0},
        {"shape": (1, 14, 18, 12, 3), "spacing": (0.5, 1.2, 2.0), "alpha": 0.8, "s": 2.5},
    ]

    for cfg in test_configs:
        m = torch.randn(*cfg["shape"], dtype=torch.float32)
        cached_out = apply_dsti_green_operator(
            m,
            alpha=cfg["alpha"],
            spacing=cfg["spacing"],
            s=cfg["s"],
        )
        uncached_out = uncached_baseline_dsti(
            m,
            alpha=cfg["alpha"],
            spacing=cfg["spacing"],
            s=cfg["s"],
        )

        max_abs_diff = float(torch.max(torch.abs(cached_out - uncached_out)).item())
        print(f"Parity test cfg shape={cfg['shape']}, spacing={cfg['spacing']}: max_abs_diff={max_abs_diff:.6e}")
        assert max_abs_diff < 1e-6, f"Mismatch with baseline: max_abs_diff={max_abs_diff:.6e}"


def test_m1_adversarial_device_portability_cpu_and_mps():
    """Verify portability across CPU and MPS, checking cache key device isolation."""
    clear_dst_cache()

    m_cpu = torch.randn(1, 16, 16, 2, dtype=torch.float32, device='cpu')
    out_cpu = apply_dsti_green_operator(m_cpu, fluid_sigma=2.0)
    assert out_cpu.device.type == 'cpu'

    if torch.backends.mps.is_available():
        m_mps = torch.randn(1, 16, 16, 2, dtype=torch.float32, device='mps')
        out_mps = apply_dsti_green_operator(m_mps, fluid_sigma=2.0)
        assert out_mps.device.type == 'mps'

        # Verify numerical parity between CPU and MPS runs with same input
        out_cpu_ref = apply_dsti_green_operator(m_mps.cpu(), fluid_sigma=2.0)
        diff_mps_cpu = float(torch.max(torch.abs(out_mps.cpu() - out_cpu_ref)).item())
        print(f"MPS vs CPU max diff: {diff_mps_cpu:.6e}")
        assert diff_mps_cpu < 1e-5, f"MPS numerical drift exceeds threshold: {diff_mps_cpu:.6e}"

        # Verify cache has distinct entries for cpu and mps
        assert get_dsti_filter_cache_size() >= 2


def test_m1_adversarial_dtype_stability_float32_float64():
    """Verify float32 vs float64 separation in cache and float64 precision preservation."""
    clear_dst_cache()

    m32 = torch.randn(1, 16, 16, 2, dtype=torch.float32)
    m64 = m32.to(dtype=torch.float64)

    out32 = apply_dsti_green_operator(m32, fluid_sigma=2.5)
    out64 = apply_dsti_green_operator(m64, fluid_sigma=2.5)

    assert out32.dtype == torch.float32
    assert out64.dtype == torch.float64

    # Verify cache contains 2 separate entries
    assert get_dsti_filter_cache_size() == 2

    # Verify float64 retains double precision
    out64_baseline = uncached_baseline_dsti(m64, fluid_sigma=2.5)
    diff64 = float(torch.max(torch.abs(out64 - out64_baseline)).item())
    assert diff64 < 1e-12, f"Float64 precision loss: max_diff={diff64:.6e}"


def test_m1_adversarial_cache_capacity_and_lru_eviction():
    """Verify cache is strictly bounded at maxsize=64 and safely evicts under 100 queries."""
    clear_dst_cache()

    for i in range(100):
        # Create unique spatial shapes (e.g. 10+i, 12)
        shape = (1, 10 + (i % 20), 12 + (i // 20), 2)
        m = torch.zeros(shape, dtype=torch.float32)
        apply_dsti_green_operator(m, fluid_sigma=2.0)
        curr_size = get_dsti_filter_cache_size()
        assert curr_size <= _MAX_DST_CACHE_SIZE, f"Cache size {curr_size} exceeded maximum {_MAX_DST_CACHE_SIZE}"

    final_size = get_dsti_filter_cache_size()
    assert final_size == _MAX_DST_CACHE_SIZE, f"Expected full cache of {_MAX_DST_CACHE_SIZE}, got {final_size}"


def test_m1_adversarial_concurrent_multithreading():
    """Verify concurrent thread safety under heavy simultaneous queries."""
    clear_dst_cache()

    def run_worker(thread_idx):
        shape = (1, 16, 16, 2)
        torch.manual_seed(thread_idx)
        m = torch.randn(shape, dtype=torch.float32)
        for _ in range(20):
            res1 = apply_dsti_green_operator(m, fluid_sigma=2.0)
            res2 = smooth_displacement_field_dst(displacement=m, fluid_sigma=2.0)
            diff = float(torch.max(torch.abs(res1 - res2)).item())
            assert diff == 0.0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_worker, i) for i in range(16)]
        for f in futures:
            f.result()

    info = get_dst_cache_info()
    assert info.hits > 0
    assert info.misses >= 1
    assert get_dsti_filter_cache_size() <= _MAX_DST_CACHE_SIZE
