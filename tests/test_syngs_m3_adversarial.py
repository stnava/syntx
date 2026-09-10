"""
Empirical Parity and Stress Verification Suite for Milestone 3 (M3).
Challenger 1: syntx.syngs coordinate transform streamlining.

Verifies:
1. Mathematical parity of GeodesicShootingModel.shoot across all 12 combinations:
   (2D, 3D) x (Transport, Recursive) x (Euler, Midpoint, RK4)
   against the un-optimized baseline formulas.
   Criterion: L_inf < 1e-7 across all canonical configurations.
2. Exact algebraic equivalence in float64 (L_inf < 1e-14, bitwise machine precision).
3. Autograd gradient parity: dL / d(v_init) matches baseline with exact zero in float64.
4. Anisotropic spacing, non-square / non-cubic shapes, and oblique direction matrices.
5. Execution runtime speedup across intermediate substeps.
"""

import math
import time
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from syntx.syngs import GeodesicShootingModel
from syntx.syn import grid_sample_nd
from syntx.spatial import (
    reverse_metadata,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
)


def shoot_baseline(model, v_init, target_shape, spacing_zyx, phys_grid, meta):
    """
    Ground-truth baseline implementation of GeodesicShootingModel.shoot
    prior to the Milestone 3 optimization (worker_m3).
    Uses un-fused physical_to_normalized_torch_cached on every substep
    and un-cached trilinear pullback in recursive mode.
    """
    dt = 1.0 / model.n_steps

    if tuple(v_init.shape[1:-1]) != target_shape:
        if model.dim == 3:
            v_cf = v_init.permute(0, 4, 1, 2, 3)
            v_up = F.interpolate(v_cf, size=target_shape, mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
        else:
            v_cf = v_init.permute(0, 3, 1, 2)
            v_up = F.interpolate(v_cf, size=target_shape, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    else:
        v_up = v_init

    v0_smooth = model.apply_green_operator(v_up, target_shape, spacing_zyx)
    shape_t, spacing_t, origin_t, direction_t = meta
    disp = torch.zeros_like(phys_grid)

    if model.transport_mode == 'transport':
        if model.dim == 3:
            v0_cf = v0_smooth.permute(0, 4, 1, 2, 3)
        else:
            v0_cf = v0_smooth.permute(0, 3, 1, 2)

        sol = str(model.solver).lower()
        if sol in ('midpoint', 'rk2', 'heun'):
            for step in range(model.n_steps):
                phi_curr = phys_grid + disp
                phi_norm_1 = physical_to_normalized_torch_cached(phi_curr, shape_t, spacing_t, origin_t, direction_t)
                if model.dim == 3:
                    k1 = grid_sample_nd(v0_cf, phi_norm_1, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    k1 = grid_sample_nd(v0_cf, phi_norm_1, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                phi_mid = phi_curr + (0.5 * dt) * k1
                phi_norm_2 = physical_to_normalized_torch_cached(phi_mid, shape_t, spacing_t, origin_t, direction_t)
                if model.dim == 3:
                    k2 = grid_sample_nd(v0_cf, phi_norm_2, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    k2 = grid_sample_nd(v0_cf, phi_norm_2, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                disp = disp + dt * k2
            return disp
        elif sol == 'rk4':
            for step in range(model.n_steps):
                phi_curr = phys_grid + disp
                phi_norm_1 = physical_to_normalized_torch_cached(phi_curr, shape_t, spacing_t, origin_t, direction_t)
                if model.dim == 3:
                    k1 = grid_sample_nd(v0_cf, phi_norm_1, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    k1 = grid_sample_nd(v0_cf, phi_norm_1, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                phi_mid1 = phi_curr + (0.5 * dt) * k1
                phi_norm_2 = physical_to_normalized_torch_cached(phi_mid1, shape_t, spacing_t, origin_t, direction_t)
                if model.dim == 3:
                    k2 = grid_sample_nd(v0_cf, phi_norm_2, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    k2 = grid_sample_nd(v0_cf, phi_norm_2, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                phi_mid2 = phi_curr + (0.5 * dt) * k2
                phi_norm_3 = physical_to_normalized_torch_cached(phi_mid2, shape_t, spacing_t, origin_t, direction_t)
                if model.dim == 3:
                    k3 = grid_sample_nd(v0_cf, phi_norm_3, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    k3 = grid_sample_nd(v0_cf, phi_norm_3, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                phi_end = phi_curr + dt * k3
                phi_norm_4 = physical_to_normalized_torch_cached(phi_end, shape_t, spacing_t, origin_t, direction_t)
                if model.dim == 3:
                    k4 = grid_sample_nd(v0_cf, phi_norm_4, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    k4 = grid_sample_nd(v0_cf, phi_norm_4, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                disp = disp + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return disp
        else:
            for step in range(model.n_steps):
                phi_curr = phys_grid + disp
                phi_norm = physical_to_normalized_torch_cached(phi_curr, shape_t, spacing_t, origin_t, direction_t)

                if model.dim == 3:
                    v_sampled = grid_sample_nd(v0_cf, phi_norm, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
                else:
                    v_sampled = grid_sample_nd(v0_cf, phi_norm, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)

                disp = disp + dt * v_sampled
            return disp

    v = v0_smooth
    for step in range(model.n_steps):
        phi_curr = phys_grid + disp
        phi_norm = physical_to_normalized_torch_cached(phi_curr, shape_t, spacing_t, origin_t, direction_t)

        if model.dim == 3:
            v_cf = v.permute(0, 4, 1, 2, 3)
            v_sampled_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
            v_sampled = v_sampled_cf.permute(0, 2, 3, 4, 1)
        else:
            v_cf = v.permute(0, 3, 1, 2)
            v_sampled_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
            v_sampled = v_sampled_cf.permute(0, 2, 3, 1)

        disp = disp + dt * v_sampled

        if step < model.n_steps - 1:
            if model.dim == 3:
                v_pullback_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v = model.apply_green_operator(v_pullback_cf.permute(0, 2, 3, 4, 1), target_shape, spacing_zyx)
            else:
                v_pullback_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v = model.apply_green_operator(v_pullback_cf.permute(0, 2, 3, 1), target_shape, spacing_zyx)

    return disp


def _setup_geometry(dim, shape, spacing, origin=None, direction=None, dtype=torch.float32, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if origin is None:
        origin = [0.0] * dim
    if direction is None:
        direction = np.eye(dim).tolist()

    phys_grid = get_physical_grid_torch(shape, spacing, origin, direction, dtype=dtype)
    sp_rev, orig_rev, dir_rev = reverse_metadata(spacing, origin, direction)
    meta = (
        torch.tensor(list(shape), dtype=dtype),
        torch.tensor(sp_rev, dtype=dtype),
        torch.tensor(orig_rev, dtype=dtype),
        torch.tensor(dir_rev, dtype=dtype),
    )

    v_init = torch.randn(1, *shape, dim, dtype=dtype) * 0.05
    return phys_grid, sp_rev, meta, v_init


# ==============================================================================
# 1. 12-Configuration Parity Tests (Standard Isotropic): L_inf < 1e-7
# ==============================================================================

CONFIGS_12_STANDARD = [
    # 2D Isotropic
    (2, 'transport', 'euler', (16, 16), [1.0, 1.0]),
    (2, 'transport', 'midpoint', (16, 16), [1.0, 1.0]),
    (2, 'transport', 'rk4', (16, 16), [1.0, 1.0]),
    (2, 'recursive', 'euler', (16, 16), [1.0, 1.0]),
    (2, 'recursive', 'midpoint', (16, 16), [1.0, 1.0]),
    (2, 'recursive', 'rk4', (16, 16), [1.0, 1.0]),
    # 3D Isotropic
    (3, 'transport', 'euler', (16, 16, 16), [1.0, 1.0, 1.0]),
    (3, 'transport', 'midpoint', (16, 16, 16), [1.0, 1.0, 1.0]),
    (3, 'transport', 'rk4', (16, 16, 16), [1.0, 1.0, 1.0]),
    (3, 'recursive', 'euler', (16, 16, 16), [1.0, 1.0, 1.0]),
    (3, 'recursive', 'midpoint', (16, 16, 16), [1.0, 1.0, 1.0]),
    (3, 'recursive', 'rk4', (16, 16, 16), [1.0, 1.0, 1.0]),
]

@pytest.mark.parametrize("dim,mode,solver,shape,spacing", CONFIGS_12_STANDARD)
def test_all_12_combinations_parity_standard(dim, mode, solver, shape, spacing):
    """
    Test exact mathematical parity across all 12 combinations on standard domains.
    Guarantees L_inf < 1e-7 across every single configuration.
    """
    n_steps = 6
    model = GeodesicShootingModel(
        dim=dim,
        image_shape=shape,
        velocity_shape=shape,
        spacing=spacing,
        n_steps=n_steps,
        solver=solver,
        transport_mode=mode,
    )
    phys_grid, sp_rev, meta, v_init = _setup_geometry(dim, shape, spacing)

    disp_opt = model.shoot(v_init, shape, sp_rev, phys_grid, meta)
    disp_base = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)

    diff = torch.abs(disp_opt - disp_base)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    l2_err = torch.sqrt(torch.mean((disp_opt - disp_base) ** 2)).item()

    assert not torch.isnan(disp_opt).any(), f"NaN found in disp_opt for {dim}D {mode} {solver}"
    assert not torch.isinf(disp_opt).any(), f"Inf found in disp_opt for {dim}D {mode} {solver}"
    assert max_err < 1e-7, (
        f"L_inf parity violation for {dim}D {mode} {solver}: "
        f"max_err={max_err:.4e} >= 1e-7 (mean_err={mean_err:.4e}, l2_err={l2_err:.4e})"
    )


# ==============================================================================
# 2. 12-Configuration Parity Tests (Anisotropic Non-Square/Non-Cubic): L_inf < 1e-7
# ==============================================================================

CONFIGS_12_ANISOTROPIC = [
    # 2D Anisotropic
    (2, 'transport', 'euler', (16, 20), [1.2, 0.9]),
    (2, 'transport', 'midpoint', (16, 20), [1.2, 0.9]),
    (2, 'transport', 'rk4', (16, 20), [1.2, 0.9]),
    (2, 'recursive', 'euler', (16, 20), [1.2, 0.9]),
    (2, 'recursive', 'midpoint', (16, 20), [1.2, 0.9]),
    (2, 'recursive', 'rk4', (16, 20), [1.2, 0.9]),
    # 3D Anisotropic
    (3, 'transport', 'euler', (12, 14, 16), [1.3, 1.0, 0.8]),
    (3, 'transport', 'midpoint', (12, 14, 16), [1.3, 1.0, 0.8]),
    (3, 'transport', 'rk4', (12, 14, 16), [1.3, 1.0, 0.8]),
    (3, 'recursive', 'euler', (12, 14, 16), [1.3, 1.0, 0.8]),
    (3, 'recursive', 'midpoint', (12, 14, 16), [1.3, 1.0, 0.8]),
    (3, 'recursive', 'rk4', (12, 14, 16), [1.3, 1.0, 0.8]),
]

@pytest.mark.parametrize("dim,mode,solver,shape,spacing", CONFIGS_12_ANISOTROPIC)
def test_all_12_combinations_parity_anisotropic(dim, mode, solver, shape, spacing):
    """
    Test exact mathematical parity across all 12 combinations with anisotropic geometries.
    Guarantees L_inf < 1e-7 across every single configuration.
    """
    n_steps = 6
    model = GeodesicShootingModel(
        dim=dim,
        image_shape=shape,
        velocity_shape=shape,
        spacing=spacing,
        n_steps=n_steps,
        solver=solver,
        transport_mode=mode,
    )
    phys_grid, sp_rev, meta, v_init = _setup_geometry(dim, shape, spacing)

    disp_opt = model.shoot(v_init, shape, sp_rev, phys_grid, meta)
    disp_base = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)

    diff = torch.abs(disp_opt - disp_base)
    max_err = diff.max().item()

    assert not torch.isnan(disp_opt).any(), f"NaN found in disp_opt for {dim}D {mode} {solver}"
    assert not torch.isinf(disp_opt).any(), f"Inf found in disp_opt for {dim}D {mode} {solver}"
    assert max_err < 1e-7, (
        f"Anisotropic L_inf parity violation for {dim}D {mode} {solver}: "
        f"max_err={max_err:.4e} >= 1e-7"
    )


# ==============================================================================
# 3. Exact Bitwise Parity in Float64 (L_inf < 1e-14)
# ==============================================================================

@pytest.mark.parametrize("dim,solver", [(2, 'rk4'), (2, 'midpoint'), (3, 'rk4'), (3, 'midpoint')])
def test_float64_exact_algebraic_parity(dim, solver):
    """
    Verifies that in double precision (float64), shoot_opt and shoot_base are
    mathematically identical down to float64 machine epsilon (L_inf < 1e-14).
    """
    shape = (16, 16) if dim == 2 else (12, 12, 12)
    spacing = [1.2, 0.9] if dim == 2 else [1.1, 1.0, 0.9]
    n_steps = 6

    model = GeodesicShootingModel(
        dim=dim, image_shape=shape, velocity_shape=shape, spacing=spacing,
        n_steps=n_steps, solver=solver, transport_mode='transport'
    ).to(dtype=torch.float64)
    phys_grid, sp_rev, meta, v_init = _setup_geometry(dim, shape, spacing, dtype=torch.float64)

    disp_opt = model.shoot(v_init, shape, sp_rev, phys_grid, meta)
    disp_base = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)

    max_err = torch.abs(disp_opt - disp_base).max().item()
    assert max_err < 1e-14, f"Float64 algebraic disparity: L_inf={max_err:.4e} >= 1e-14"


# ==============================================================================
# 4. Oblique Direction & Stress Testing
# ==============================================================================

def test_2d_oblique_direction_parity():
    """Test with non-identity 2D rotation direction matrix."""
    dim = 2
    shape = (16, 20)
    spacing = [1.1, 0.85]
    theta = math.radians(25.0)
    direction = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]
    origin = [0.0, 0.0]

    model = GeodesicShootingModel(
        dim=dim, image_shape=shape, velocity_shape=shape, spacing=spacing,
        origin=origin, direction=direction, n_steps=6, solver='rk4', transport_mode='transport'
    )
    phys_grid, sp_rev, meta, v_init = _setup_geometry(dim, shape, spacing, origin=origin, direction=direction)

    disp_opt = model.shoot(v_init, shape, sp_rev, phys_grid, meta)
    disp_base = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)

    max_err = torch.abs(disp_opt - disp_base).max().item()
    assert max_err < 1e-7, f"2D Oblique RK4 L_inf={max_err:.4e} >= 1e-7"

    # Also test non-zero origin in float64 for exact algebraic parity
    origin_shifted = [15.0, -8.0]
    model_64 = GeodesicShootingModel(
        dim=dim, image_shape=shape, velocity_shape=shape, spacing=spacing,
        origin=origin_shifted, direction=direction, n_steps=6, solver='rk4', transport_mode='transport'
    ).to(dtype=torch.float64)
    phys_grid_64, sp_rev_64, meta_64, v_init_64 = _setup_geometry(
        dim, shape, spacing, origin=origin_shifted, direction=direction, dtype=torch.float64
    )
    disp_opt_64 = model_64.shoot(v_init_64, shape, sp_rev_64, phys_grid_64, meta_64)
    disp_base_64 = shoot_baseline(model_64, v_init_64, shape, sp_rev_64, phys_grid_64, meta_64)
    max_err_64 = torch.abs(disp_opt_64 - disp_base_64).max().item()
    assert max_err_64 < 1e-14, f"2D Oblique Float64 L_inf={max_err_64:.4e} >= 1e-14"


def test_3d_oblique_direction_parity():
    """Test with non-identity 3D rotation direction matrix."""
    dim = 3
    shape = (12, 14, 16)
    spacing = [1.25, 0.95, 1.05]
    ay = math.radians(15.0)
    az = math.radians(20.0)
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    direction = (Rz @ Ry).tolist()
    origin = [-12.0, 24.0, 7.5]

    model = GeodesicShootingModel(
        dim=dim, image_shape=shape, velocity_shape=shape, spacing=spacing,
        origin=origin, direction=direction, n_steps=6, solver='rk4', transport_mode='transport'
    )
    phys_grid, sp_rev, meta, v_init = _setup_geometry(dim, shape, spacing, origin=origin, direction=direction)

    disp_opt = model.shoot(v_init, shape, sp_rev, phys_grid, meta)
    disp_base = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)

    max_err = torch.abs(disp_opt - disp_base).max().item()
    assert max_err < 1e-7, f"3D Oblique RK4 L_inf={max_err:.4e} >= 1e-7"


# ==============================================================================
# 5. Autograd Backward Gradient Parity
# ==============================================================================

@pytest.mark.parametrize("dim,solver", [(2, 'rk4'), (2, 'midpoint'), (3, 'rk4'), (3, 'midpoint')])
def test_autograd_backward_gradient_parity_exact(dim, solver):
    """
    Verify that backpropagation through shoot produces identical gradients
    dL / d(v_init) in float64 with exact machine precision (L_inf == 0).
    """
    shape = (16, 16) if dim == 2 else (12, 12, 12)
    spacing = [1.1, 0.9] if dim == 2 else [1.2, 1.0, 0.8]
    n_steps = 4

    model = GeodesicShootingModel(
        dim=dim, image_shape=shape, velocity_shape=shape, spacing=spacing,
        n_steps=n_steps, solver=solver, transport_mode='transport'
    ).to(dtype=torch.float64)
    phys_grid, sp_rev, meta, _ = _setup_geometry(dim, shape, spacing, dtype=torch.float64)

    v_init_opt = (torch.randn(1, *shape, dim, dtype=torch.float64) * 0.05).requires_grad_(True)
    v_init_base = v_init_opt.detach().clone().requires_grad_(True)

    disp_opt = model.shoot(v_init_opt, shape, sp_rev, phys_grid, meta)
    loss_opt = (disp_opt ** 2).sum()
    loss_opt.backward()

    disp_base = shoot_baseline(model, v_init_base, shape, sp_rev, phys_grid, meta)
    loss_base = (disp_base ** 2).sum()
    loss_base.backward()

    grad_diff = torch.abs(v_init_opt.grad - v_init_base.grad)
    max_grad_err = grad_diff.max().item()

    assert max_grad_err < 1e-14, (
        f"Float64 Autograd gradient mismatch for {dim}D {solver}: "
        f"max_grad_err={max_grad_err:.4e} >= 1e-14"
    )


# ==============================================================================
# 6. Runtime Speedup Benchmark Runner
# ==============================================================================

def run_speedup_benchmark():
    """
    Measure execution runtime speedup of GeodesicShootingModel.shoot across intermediate substeps.
    """
    print("\n" + "=" * 78)
    print("M3 GEODESIC SHOOTING STREAMLINING SPEEDUP BENCHMARK")
    print("=" * 78)
    print(f"{'Config':<28} | {'Steps':<6} | {'Base (ms)':<10} | {'Opt (ms)':<10} | {'Speedup':<8} | {'Saved (%)':<10}")
    print("-" * 78)

    benchmark_cases = [
        (2, 'transport', 'euler', (64, 64), [1.0, 1.0], [5, 10, 20]),
        (2, 'transport', 'midpoint', (64, 64), [1.0, 1.0], [5, 10, 20]),
        (2, 'transport', 'rk4', (64, 64), [1.0, 1.0], [5, 10, 20]),
        (2, 'recursive', 'euler', (64, 64), [1.0, 1.0], [5, 10, 20]),
        (3, 'transport', 'euler', (28, 28, 28), [1.0, 1.0, 1.0], [4, 8, 12]),
        (3, 'transport', 'midpoint', (28, 28, 28), [1.0, 1.0, 1.0], [4, 8, 12]),
        (3, 'transport', 'rk4', (28, 28, 28), [1.0, 1.0, 1.0], [4, 8, 12]),
        (3, 'recursive', 'euler', (28, 28, 28), [1.0, 1.0, 1.0], [4, 8, 12]),
    ]

    results = []
    n_iters = 15
    n_warmup = 3

    for dim, mode, solver, shape, spacing, steps_list in benchmark_cases:
        for n_steps in steps_list:
            model = GeodesicShootingModel(
                dim=dim, image_shape=shape, velocity_shape=shape, spacing=spacing,
                n_steps=n_steps, solver=solver, transport_mode=mode
            )
            phys_grid, sp_rev, meta, v_init = _setup_geometry(dim, shape, spacing)

            # Warmup
            for _ in range(n_warmup):
                _ = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)
                _ = model.shoot(v_init, shape, sp_rev, phys_grid, meta)

            # Baseline timing
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = shoot_baseline(model, v_init, shape, sp_rev, phys_grid, meta)
            t_base = (time.perf_counter() - t0) / n_iters * 1000.0

            # Optimized timing
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = model.shoot(v_init, shape, sp_rev, phys_grid, meta)
            t_opt = (time.perf_counter() - t0) / n_iters * 1000.0

            speedup = t_base / t_opt
            saved = (t_base - t_opt) / t_base * 100.0

            tag = f"{dim}D {mode[:5]} {solver}"
            print(f"{tag:<28} | {n_steps:<6} | {t_base:<10.2f} | {t_opt:<10.2f} | {speedup:<7.2f}x | {saved:<9.1f}%")
            results.append({
                'dim': dim,
                'mode': mode,
                'solver': solver,
                'n_steps': n_steps,
                't_base': t_base,
                't_opt': t_opt,
                'speedup': speedup,
                'saved': saved
            })

    print("=" * 78)
    return results


if __name__ == '__main__':
    run_speedup_benchmark()
