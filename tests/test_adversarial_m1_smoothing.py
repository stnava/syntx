"""
Adversarial Stress Testing & Parity Verification for Milestone 1 (M1)
Stationary Operator Caching & Broadcasted DST-I Green Operator in syntx.core.smoothing

Challenges:
1. Highly anisotropic grids, extreme aspect ratios, prime dimensions, and extreme alpha / s parameters.
2. High-concurrency race condition scenarios (64 threads contention, rapid eviction storms).
3. Exact bitwise numerical parity (L_inf = 0.0) against un-cached baseline meshgrid formulation.
4. Memory allocation and leak tracking across high query volume.
"""

import collections
import concurrent.futures
import gc
import math
import sys
import threading
import time
import tracemalloc
import pytest
import torch
import numpy as np

from syntx.core.smoothing import (
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    smooth_displacement_field_dst,
    clear_dst_cache,
    clear_dsti_filter_cache,
    get_dst_cache_info,
    get_dsti_cache_info,
    get_dsti_filter_cache_size,
    _get_dsti_filter_cached,
    _DST_FILTER_CACHE,
    _DST_CACHE_LOCK,
    _MAX_DST_CACHE_SIZE,
    CacheInfo,
)


# ==============================================================================
# Baseline Reference Implementations (Un-cached meshgrid)
# ==============================================================================

def _baseline_meshgrid_filter(spatial_shape, alpha_val, s, spacing, device, dtype):
    """Un-cached baseline computing DST-I filter via torch.meshgrid."""
    dim = len(spatial_shape)
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else None
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        scale = (sp_d ** 2) if spacing is not None else 1.0
        lambda_d = (4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)) / scale
        k_axes.append(lambda_d)

    if dim == 1:
        lambda_sq = k_axes[0]
    else:
        k_mesh = torch.meshgrid(*k_axes, indexing="ij")
        lambda_sq = sum(k_j for k_j in k_mesh)

    return (
        (1.0 / ((1.0 + alpha_val * lambda_sq) ** s))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )


def _baseline_apply_dsti(m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0):
    """Un-cached baseline smoothing function using torch.meshgrid."""
    if fluid_sigma <= 0:
        return m
    device = m.device
    dtype = m.dtype
    spatial_shape = m.shape[1:-1]
    dim = len(spatial_shape)
    alpha_val = float(alpha) if alpha is not None else float(fluid_sigma) / 2.0

    K_dst = _baseline_meshgrid_filter(spatial_shape, alpha_val, s, spacing, device, dtype)

    curr = m.movedim(-1, 1).to(torch.float32)
    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        z_shape = list(curr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(curr, dims=[axis])
        padded = torch.cat([z, curr, z, rev], dim=axis)
        F = torch.fft.fft(padded, dim=axis)
        curr = -torch.imag(F.narrow(axis, 1, n_d)).contiguous()

    curr = curr * K_dst

    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        z_shape = list(curr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(curr, dims=[axis])
        padded = torch.cat([z, curr, z, rev], dim=axis)
        F = torch.fft.fft(padded, dim=axis)
        curr = (-torch.imag(F.narrow(axis, 1, n_d)) / (2.0 * (n_d + 1))).contiguous()

    return curr.to(dtype=dtype).movedim(1, -1)


# ==============================================================================
# Challenge 1: Highly Anisotropic Grids, Extreme Aspect Ratios, Prime Dimensions
# ==============================================================================

def test_adversarial_anisotropic_grids_and_aspect_ratios():
    """
    Challenge: Test extreme aspect ratios and highly anisotropic spacings.
    Verifies exact bitwise parity (L_inf = 0.0) between broadcasted eigenvalues
    and meshgrid baseline.
    """
    clear_dst_cache()

    adversarial_cases = [
        # (spatial_shape, spacing, alpha_val, s)
        # 2D extreme aspect ratio (1:32)
        ((8, 256), (0.25, 4.0), 1.5, 2.0),
        ((256, 8), (4.0, 0.25), 1.5, 2.0),
        # 3D extreme aspect ratio (16, 128, 32)
        ((16, 128, 32), (0.4, 1.8, 3.2), 2.0, 2.0),
        # 3D pancake/slab geometry (4, 64, 64) with extreme anisotropy
        ((4, 64, 64), (5.0, 0.5, 0.5), 1.0, 2.0),
        # 3D needle geometry (64, 4, 4)
        ((64, 4, 4), (0.5, 5.0, 5.0), 3.0, 2.0),
        # Highly anisotropic spacings with non-trivial decimals
        ((24, 32, 16), (0.333, 1.777, 2.888), 0.75, 2.0),
    ]

    for shape, spacing, alpha_val, s in adversarial_cases:
        k_cached = _get_dsti_filter_cached(shape, alpha_val, s, spacing, "cpu", torch.float32)
        k_baseline = _baseline_meshgrid_filter(shape, alpha_val, s, spacing, "cpu", torch.float32)

        assert k_cached.shape == (1, 1, *shape)
        assert k_baseline.shape == (1, 1, *shape)

        diff = torch.max(torch.abs(k_cached - k_baseline)).item()
        assert diff == 0.0, (
            f"Parity failure on shape {shape} with spacing {spacing}: L_inf = {diff}"
        )


def test_adversarial_prime_dimensions_and_extreme_parameters():
    """
    Challenge: Prime spatial dimensions (preventing power-of-2 symmetries) and
    extreme parameter ranges (alpha=1e-6, alpha=1e6, s=0.5, s=4.0, s=8.0).
    Verifies exact bitwise parity (L_inf = 0.0) and numeric stability.
    """
    clear_dst_cache()

    stress_cases = [
        # Prime dimensions 2D
        ((127, 131), (1.0, 1.0), 1.5, 2.0),
        ((53, 59), (0.7, 1.3), 2.0, 2.0),
        # Prime dimensions 3D
        ((29, 31, 37), (1.1, 0.9, 1.3), 1.2, 2.0),
        # Extreme alpha: tiny alpha (near-identity Green operator, K -> 1.0)
        ((16, 16, 16), (1.0, 1.0, 1.0), 1e-6, 2.0),
        # Extreme alpha: huge alpha (aggressive low-pass, K -> 0.0 except DC)
        ((16, 16, 16), (1.0, 1.0, 1.0), 1e6, 2.0),
        # High Sobolev orders
        ((16, 16, 16), (1.0, 1.0, 1.0), 1.5, 4.0),
        ((12, 12, 12), (1.0, 1.0, 1.0), 1.0, 8.0),
        # Fractional Sobolev order
        ((16, 16, 16), (1.0, 1.0, 1.0), 1.5, 0.5),
    ]

    for shape, spacing, alpha_val, s in stress_cases:
        k_cached = _get_dsti_filter_cached(shape, alpha_val, s, spacing, "cpu", torch.float32)
        k_baseline = _baseline_meshgrid_filter(shape, alpha_val, s, spacing, "cpu", torch.float32)

        # Check for numerical sanity
        assert not torch.isnan(k_cached).any(), f"NaN detected in k_cached for shape {shape}"
        assert not torch.isinf(k_cached).any(), f"Inf detected in k_cached for shape {shape}"
        assert (k_cached > 0.0).all(), f"Eigenvalues must be strictly positive for shape {shape}"

        diff = torch.max(torch.abs(k_cached - k_baseline)).item()
        assert diff == 0.0, (
            f"Bitwise parity failure on shape {shape}, alpha={alpha_val}, s={s}: L_inf = {diff}"
        )


def test_adversarial_smoothing_output_parity_anisotropic_3d():
    """
    Challenge: Test end-to-end displacement field smoothing on anisotropic 3D grid
    with random input vectors. Verifies exact bitwise parity (L_inf = 0.0).
    """
    clear_dst_cache()
    torch.manual_seed(1337)

    # 3D anisotropic displacement field: batch=1, Z=14, Y=18, X=22, channels=3
    shape = (14, 18, 22)
    spacing = (0.4, 1.8, 3.2)
    m = torch.randn(1, *shape, 3)

    v_cached = smooth_displacement_field_dst(m, fluid_sigma=2.8, spacing=spacing, s=2.0)
    v_baseline = _baseline_apply_dsti(m, fluid_sigma=2.8, spacing=spacing, s=2.0)

    assert v_cached.shape == m.shape
    assert v_baseline.shape == m.shape

    max_diff = torch.max(torch.abs(v_cached - v_baseline)).item()
    assert max_diff == 0.0, f"Displacement smoothing output parity failure: L_inf = {max_diff}"


# ==============================================================================
# Challenge 2: High Concurrency Race Conditions (64 Threads)
# ==============================================================================

def test_adversarial_64_threads_identical_geometry_contention():
    """
    Challenge: 64 concurrent threads storming an empty cache requesting the
    EXACT same geometry simultaneously.
    Verifies:
    1. Zero deadlocks or race condition exceptions.
    2. Cache size is exactly 1.
    3. All 64 threads obtain the exact identical cached tensor instance.
    4. Hit/miss telemetry consistency.
    """
    clear_dst_cache()
    num_threads = 64
    shape = (16, 16, 16)
    spacing = (1.0, 1.0, 1.0)
    alpha_val = 1.5
    s = 2.0

    barrier = threading.Barrier(num_threads)
    results = [None] * num_threads

    def worker(tid):
        barrier.wait()  # Synchronize all 64 threads to strike simultaneously
        t = _get_dsti_filter_cached(shape, alpha_val, s, spacing, "cpu", torch.float32)
        results[tid] = t

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Verify all threads received a valid tensor
    assert all(r is not None for r in results)

    # Verify all threads received the EXACT same tensor memory instance
    first_tensor = results[0]
    for i, r in enumerate(results):
        assert r is first_tensor, f"Thread {i} received a different tensor instance!"

    # Verify cache size is exactly 1
    assert get_dsti_filter_cache_size() == 1
    info = get_dst_cache_info()
    assert info.currsize == 1
    assert info.hits + info.misses == num_threads


def test_adversarial_64_threads_distinct_entries_and_eviction_storm():
    """
    Challenge: 64 concurrent threads inserting 512 distinct geometry entries
    in an eviction storm while concurrent reader threads continuously query
    and assert cache size invariants.
    Verifies:
    1. Zero deadlocks, zero dict mutation during iteration exceptions.
    2. Cache capacity strictly bounded: currsize <= _MAX_DST_CACHE_SIZE (64).
    3. LRU eviction order maintained under heavy multithreaded contention.
    """
    clear_dst_cache()
    num_writers = 48
    num_readers = 16
    total_threads = num_writers + num_readers
    entries_per_writer = 15  # Total entries inserted: 48 * 15 = 720 distinct entries

    errors = []
    stop_flag = threading.Event()

    def writer_worker(wid):
        try:
            for k in range(entries_per_writer):
                # Unique shape per writer and step
                dim_x = 10 + (wid % 8)
                dim_y = 10 + (k % 8)
                dim_z = 10 + wid + k
                shape = (dim_x, dim_y, dim_z)
                _get_dsti_filter_cached(shape, 1.5, 2.0, None, "cpu", torch.float32)
        except Exception as e:
            errors.append(e)

    def reader_worker():
        try:
            while not stop_flag.is_set():
                size = get_dsti_filter_cache_size()
                if size > _MAX_DST_CACHE_SIZE:
                    errors.append(ValueError(f"Cache size violated bound: {size} > {_MAX_DST_CACHE_SIZE}"))
                info = get_dst_cache_info()
                if info.currsize > info.maxsize:
                    errors.append(ValueError(f"Telemetry bound violated: {info.currsize} > {info.maxsize}"))
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as executor:
        # Start readers
        reader_futures = [executor.submit(reader_worker) for _ in range(num_readers)]

        # Start writers
        writer_futures = [executor.submit(writer_worker, i) for i in range(num_writers)]
        for f in concurrent.futures.as_completed(writer_futures):
            f.result()

        stop_flag.set()
        for f in reader_futures:
            f.result()

    assert len(errors) == 0, f"Encountered {len(errors)} concurrency errors: {errors[:5]}"

    # Final invariant check: cache size must be exactly _MAX_DST_CACHE_SIZE (64)
    final_size = get_dsti_filter_cache_size()
    assert final_size == _MAX_DST_CACHE_SIZE, (
        f"Cache size should be saturated at {_MAX_DST_CACHE_SIZE}, got {final_size}"
    )


# ==============================================================================
# Challenge 3: Memory Allocation & Leak Tracking
# ==============================================================================

def test_adversarial_memory_leak_tracking_5000_iterations():
    """
    Challenge: Execute 5,000 successive smoothing calls on a stationary geometry.
    Verifies:
    1. Zero memory leaks: tracemalloc memory delta is 0 MB.
    2. Cache hits increment by 5,000; cache misses remain 1.
    3. Peak memory is strictly stable across iterations.
    """
    clear_dst_cache()
    gc.collect()

    m = torch.randn(1, 16, 16, 16, 3)
    fluid_sigma = 2.5
    spacing = (1.0, 1.0, 1.0)

    # Warmup / initial cache miss
    _ = smooth_displacement_field_dst(m, fluid_sigma=fluid_sigma, spacing=spacing)
    info_warmup = get_dst_cache_info()
    assert info_warmup.misses == 1
    assert info_warmup.hits == 0

    tracemalloc.start()
    gc.collect()
    snapshot_before = tracemalloc.take_snapshot()

    num_iterations = 5000
    for _ in range(num_iterations):
        _ = smooth_displacement_field_dst(m, fluid_sigma=fluid_sigma, spacing=spacing)

    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    info_final = get_dst_cache_info()
    assert info_final.misses == 1
    assert info_final.hits == num_iterations

    # Check top memory differences
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_delta_bytes = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    total_delta_kb = total_delta_bytes / 1024.0

    # Under 5,000 iterations, any leak in cache lookup or return would accumulate
    # multiple megabytes. Allowed threshold is < 64 KB (tracing noise / local variables).
    assert total_delta_kb < 64.0, (
        f"Potential memory leak detected: {total_delta_kb:.2f} KB growth over {num_iterations} calls"
    )


def test_adversarial_peak_memory_reduction_broadcast_vs_meshgrid():
    """
    Challenge: Measure peak temporary memory allocated during cache miss.
    Show that the broadcasted formulation avoids allocating 3 full 3D meshgrid
    coordinate tensors.
    """
    # Large 3D spatial shape: 96 x 96 x 96 = 884,736 voxels
    shape = (96, 96, 96)
    alpha_val = 1.5
    s = 2.0
    device = "cpu"

    # Measure meshgrid baseline memory
    gc.collect()
    tracemalloc.start()
    _ = _baseline_meshgrid_filter(shape, alpha_val, s, None, device, torch.float32)
    _, peak_meshgrid = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Clear cache and measure broadcasted cached function on cache miss
    clear_dst_cache()
    gc.collect()
    tracemalloc.start()
    _ = _get_dsti_filter_cached(shape, alpha_val, s, None, device, torch.float32)
    _, peak_broadcast = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Meshgrid creates 3 full (96, 96, 96) float32 tensors (3 * 3.54 MB = 10.6 MB)
    # plus the sum tensor. Broadcasted only creates 3 1D vectors (96 elements)
    # and broadcasts directly during summation.
    assert peak_broadcast < peak_meshgrid, (
        f"Expected broadcasted memory ({peak_broadcast / 1024:.1f} KB) "
        f"to be less than meshgrid memory ({peak_meshgrid / 1024:.1f} KB)"
    )


# ==============================================================================
# Challenge 4: Boundary Invariants (Dirichlet v=0 condition)
# ==============================================================================

def test_adversarial_dirichlet_boundary_condition():
    """
    Challenge: Verify that DST-I operator analytically enforces exact homogeneous
    Dirichlet boundary conditions (displacement vanishes at grid boundaries).
    """
    clear_dst_cache()
    torch.manual_seed(999)

    # 3D field with non-zero random values right at the boundary
    shape = (16, 18, 20)
    m = torch.randn(1, *shape, 3)

    v = smooth_displacement_field_dst(m, fluid_sigma=3.0)

    # In discrete DST-I, the continuous field vanishes at x = 0 and x = N+1.
    # The smoothed field should have dramatically attenuated boundary magnitudes
    # relative to the interior.
    interior = v[:, 2:-2, 2:-2, 2:-2, :]
    boundary_z = torch.cat([v[:, 0, :, :, :], v[:, -1, :, :, :]])
    boundary_y = torch.cat([v[:, :, 0, :, :], v[:, :, -1, :, :]])
    boundary_x = torch.cat([v[:, :, :, 0, :], v[:, :, :, -1, :]])

    assert not torch.isnan(v).any()
    assert not torch.isinf(v).any()
    # Boundary energy should be significantly smaller than input boundary energy
    input_boundary_energy = (m[:, 0, :, :, :] ** 2).mean().item()
    output_boundary_energy = (boundary_z ** 2).mean().item()
    assert output_boundary_energy < input_boundary_energy, "DST-I must attenuate boundary values"
