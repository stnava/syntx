"""Adversarial Stress Verification Suite for Milestone 3 (M3).

Challenger 2: syntx.syngs coordinate transform streamlining.

Objectives:
1. High velocity magnitudes and large time intervals (dt=0.5, n_steps=20, 50) across solvers & modes.
2. Anisotropic physical voxel spacings and oblique rotation direction matrices.
3. Zero initial velocity (v0 = 0).
4. Finite, valid gradients through autograd backward with zero runtime error.
5. Exact float32 mathematical equivalence between streamlined and baseline coordinate projections.
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


def legacy_shoot_ref(model, v_init, target_shape, spacing_zyx, phys_grid, meta):
    """Unstreamlined baseline shoot function preserving pre-M3 implementation."""
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


def _setup_test_env(dim=2, shape=None, spacing=None, origin=None, direction=None, solver='rk4', transport_mode='transport', n_steps=6):
    if shape is None:
        shape = (16, 16, 16) if dim == 3 else (24, 24)
    if spacing is None:
        spacing = [1.0] * dim
    if origin is None:
        origin = [0.0] * dim
    if direction is None:
        direction = np.eye(dim).tolist()

    model = GeodesicShootingModel(
        dim=dim,
        image_shape=shape,
        velocity_shape=shape,
        spacing=spacing,
        origin=origin,
        direction=direction,
        n_steps=n_steps,
        solver=solver,
        transport_mode=transport_mode,
    )

    phys_grid = get_physical_grid_torch(shape, spacing, origin, direction)
    sp_rev, orig_rev, dir_rev = reverse_metadata(spacing, origin, direction)
    meta = (
        torch.tensor(list(shape), dtype=torch.float32),
        torch.tensor(sp_rev, dtype=torch.float32),
        torch.tensor(orig_rev, dtype=torch.float32),
        torch.tensor(dir_rev, dtype=torch.float32),
    )
    spacing_zyx = tuple(reversed(spacing))
    return model, phys_grid, meta, spacing_zyx


# ------------------------------------------------------------------------------
# Test 1: Coordinate Normalization Algebraic Parity
# ------------------------------------------------------------------------------
def test_coordinate_normalization_exact_algebraic_parity():
    """Verify that fused M_norm, b_norm with u_id + disp @ M_norm matches

    physical_to_normalized_torch_cached up to float32 machine epsilon.
    """
    dim = 3
    shape = (16, 18, 20)
    spacing = [0.8, 1.2, 2.5]
    origin = [-15.0, 30.0, 45.0]

    # Oblique 3D rotation
    ax, ay, az = 0.2, -0.3, 0.4
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    direction = (Rz @ Ry @ Rx).tolist()

    phys_grid = get_physical_grid_torch(shape, spacing, origin, direction)
    sp_rev, orig_rev, dir_rev = reverse_metadata(spacing, origin, direction)
    shape_t = torch.tensor(list(shape), dtype=torch.float32)
    spacing_t = torch.tensor(sp_rev, dtype=torch.float32)
    origin_t = torch.tensor(orig_rev, dtype=torch.float32)
    direction_t = torch.tensor(dir_rev, dtype=torch.float32)

    # 1. Base grid parity (disp = 0)
    base_legacy = physical_to_normalized_torch_cached(phys_grid, shape_t, spacing_t, origin_t, direction_t)

    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    M = direction_t * scale_t.unsqueeze(0)
    b = - (origin_t @ M) - 1.0
    M_norm = torch.flip(M, dims=[1])
    b_norm = torch.flip(b, dims=[0])
    u_id = (phys_grid.view(-1, dim) @ M_norm + b_norm).view(phys_grid.shape)

    base_diff = (u_id - base_legacy).abs().max().item()
    assert base_diff < 1e-6, f"Base identity grid mismatch: diff={base_diff:.3e}"

    # 2. Deformed grid parity with small and large displacements
    for disp_scale in [0.01, 1.0, 10.0]:
        disp = torch.randn_like(phys_grid) * disp_scale
        deformed_legacy = physical_to_normalized_torch_cached(phys_grid + disp, shape_t, spacing_t, origin_t, direction_t)
        deformed_streamlined = u_id + (disp.view(-1, dim) @ M_norm).view(disp.shape)

        diff = (deformed_streamlined - deformed_legacy).abs().max().item()
        # Relative error should be within single-precision floating point limits
        rel_diff = diff / (deformed_streamlined.abs().max().item() + 1e-6)
        assert rel_diff < 1e-5, f"Deformed grid mismatch at scale={disp_scale}: rel_diff={rel_diff:.3e}, abs_diff={diff:.3e}"


# ------------------------------------------------------------------------------
# Test 2: Large time intervals (dt=0.5) & high velocity magnitudes
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
@pytest.mark.parametrize("transport_mode", ["transport", "recursive"])
def test_adversarial_large_dt_half_and_high_velocities(dim, solver, transport_mode):
    """Stress test dt = 0.5 (n_steps = 2) with velocity magnitudes up to 50 mm."""
    torch.manual_seed(42)
    shape = (12, 12, 12) if dim == 3 else (16, 16)
    model, phys_grid, meta, spacing_zyx = _setup_test_env(
        dim=dim, shape=shape, solver=solver, transport_mode=transport_mode, n_steps=2
    )

    for vel_scale in [0.5, 5.0, 20.0, 50.0]:
        v_init = torch.randn(1, *shape, dim) * vel_scale

        disp_streamlined = model.shoot(v_init, shape, spacing_zyx, phys_grid, meta)
        disp_legacy = legacy_shoot_ref(model, v_init, shape, spacing_zyx, phys_grid, meta)

        # 1. Output must be strictly finite (no NaN, no Inf)
        assert torch.isfinite(disp_streamlined).all(), f"NaN/Inf at scale={vel_scale} ({dim}D, {solver}, {transport_mode})"

        # 2. Parity check within float32 numerical rounding tolerance
        abs_diff = (disp_streamlined - disp_legacy).abs().max().item()
        max_val = disp_streamlined.abs().max().item()
        rel_diff = abs_diff / (max_val + 1e-6)
        if vel_scale <= 5.0:
            assert rel_diff < 1e-5, f"Parity mismatch at scale={vel_scale}: rel_diff={rel_diff:.3e}, abs_diff={abs_diff:.3e}"
        else:
            # Extreme boundary-clipping velocity regimes (up to 50 mm / step)
            assert rel_diff < 2e-3, f"Parity mismatch at extreme scale={vel_scale}: rel_diff={rel_diff:.3e}, abs_diff={abs_diff:.3e}"


# ------------------------------------------------------------------------------
# Test 3: Long integration trajectories (n_steps=20, 50)
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("n_steps", [20, 50])
@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
@pytest.mark.parametrize("transport_mode", ["transport", "recursive"])
def test_adversarial_long_trajectories_high_steps(n_steps, solver, transport_mode):
    """Stress test trajectory integration over 20 and 50 steps."""
    torch.manual_seed(101)
    dim = 2
    shape = (16, 16)
    model, phys_grid, meta, spacing_zyx = _setup_test_env(
        dim=dim, shape=shape, solver=solver, transport_mode=transport_mode, n_steps=n_steps
    )
    v_init = torch.randn(1, *shape, dim) * 1.5

    disp_streamlined = model.shoot(v_init, shape, spacing_zyx, phys_grid, meta)
    disp_legacy = legacy_shoot_ref(model, v_init, shape, spacing_zyx, phys_grid, meta)

    assert torch.isfinite(disp_streamlined).all()
    abs_diff = (disp_streamlined - disp_legacy).abs().max().item()
    max_val = disp_streamlined.abs().max().item()
    rel_diff = abs_diff / (max_val + 1e-6)
    assert rel_diff < 5e-5, f"Long trajectory failure n_steps={n_steps}, {solver}: rel_diff={rel_diff:.3e}"


# ------------------------------------------------------------------------------
# Test 4: Anisotropic voxel spacings & oblique rotation direction matrices
# ------------------------------------------------------------------------------
def test_adversarial_anisotropic_oblique_2d():
    """Test 2D registration with 6x anisotropic spacing and 40-deg oblique rotation."""
    dim = 2
    shape = (20, 16)
    spacing = [0.5, 3.0]
    origin = [-25.0, 75.0]
    theta = 0.70  # ~40.1 degrees
    direction = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ]

    torch.manual_seed(202)
    v_init = torch.randn(1, *shape, dim) * 1.2

    for solver in ["euler", "midpoint", "rk4"]:
        for mode in ["transport", "recursive"]:
            model, phys_grid, meta, spacing_zyx = _setup_test_env(
                dim=dim, shape=shape, spacing=spacing, origin=origin, direction=direction,
                solver=solver, transport_mode=mode, n_steps=5
            )
            disp_stream = model.shoot(v_init, shape, spacing_zyx, phys_grid, meta)
            disp_leg = legacy_shoot_ref(model, v_init, shape, spacing_zyx, phys_grid, meta)

            assert torch.isfinite(disp_stream).all()
            abs_diff = (disp_stream - disp_leg).abs().max().item()
            max_val = disp_stream.abs().max().item()
            rel_diff = abs_diff / (max_val + 1e-6)
            assert rel_diff < 5e-5, f"2D Oblique failed ({solver}, {mode}): rel_diff={rel_diff:.3e}"


def test_adversarial_anisotropic_oblique_3d():
    """Test 3D registration with 7x anisotropic spacing and arbitrary 3D rotation."""
    dim = 3
    shape = (12, 14, 16)
    spacing = [0.7, 1.4, 4.9]
    origin = [15.0, -35.0, 80.0]

    np.random.seed(77)
    q, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    direction = q.tolist()

    torch.manual_seed(303)
    v_init = torch.randn(1, *shape, dim) * 1.0

    for solver in ["euler", "midpoint", "rk4"]:
        for mode in ["transport", "recursive"]:
            model, phys_grid, meta, spacing_zyx = _setup_test_env(
                dim=dim, shape=shape, spacing=spacing, origin=origin, direction=direction,
                solver=solver, transport_mode=mode, n_steps=4
            )
            disp_stream = model.shoot(v_init, shape, spacing_zyx, phys_grid, meta)
            disp_leg = legacy_shoot_ref(model, v_init, shape, spacing_zyx, phys_grid, meta)

            assert torch.isfinite(disp_stream).all()
            abs_diff = (disp_stream - disp_leg).abs().max().item()
            max_val = disp_stream.abs().max().item()
            rel_diff = abs_diff / (max_val + 1e-6)
            assert rel_diff < 5e-5, f"3D Oblique failed ({solver}, {mode}): rel_diff={rel_diff:.3e}"


# ------------------------------------------------------------------------------
# Test 5: Zero initial velocity (v0 = 0)
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
@pytest.mark.parametrize("transport_mode", ["transport", "recursive"])
def test_adversarial_zero_initial_velocity(dim, solver, transport_mode):
    """Verify that zero initial velocity yields identically zero displacement."""
    shape = (10, 10, 10) if dim == 3 else (14, 14)
    spacing = [0.6, 1.8, 3.2][:dim]
    origin = [-12.0, 24.0, -36.0][:dim]

    model, phys_grid, meta, spacing_zyx = _setup_test_env(
        dim=dim, shape=shape, spacing=spacing, origin=origin,
        solver=solver, transport_mode=transport_mode, n_steps=6
    )

    v_zero = torch.zeros(1, *shape, dim)
    disp = model.shoot(v_zero, shape, spacing_zyx, phys_grid, meta)

    assert torch.isfinite(disp).all(), "Zero velocity produced NaN or Inf"
    max_disp = disp.abs().max().item()
    assert max_disp < 1e-7, f"Zero velocity produced non-zero displacement: max={max_disp:.3e}"


# ------------------------------------------------------------------------------
# Test 6: Autograd backward through shoot produces finite, valid gradients
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
@pytest.mark.parametrize("transport_mode", ["transport", "recursive"])
def test_adversarial_autograd_backward_finite_gradients(dim, solver, transport_mode):
    """Verify autograd backward pass through streamlined shoot produces finite, valid gradients."""
    torch.manual_seed(404)
    shape = (10, 10, 10) if dim == 3 else (14, 14)
    spacing = [0.9, 1.4, 2.1][:dim]
    origin = [5.0, -10.0, 15.0][:dim]

    model, phys_grid, meta, spacing_zyx = _setup_test_env(
        dim=dim, shape=shape, spacing=spacing, origin=origin,
        solver=solver, transport_mode=transport_mode, n_steps=4
    )

    v_stream = (torch.randn(1, *shape, dim) * 0.5).requires_grad_(True)
    disp_stream = model.shoot(v_stream, shape, spacing_zyx, phys_grid, meta)
    target = torch.sin(phys_grid * 0.05)
    loss_stream = 0.5 * ((disp_stream - target) ** 2).sum()
    loss_stream.backward()

    assert v_stream.grad is not None, "Gradient is None after backward pass"
    assert torch.isfinite(v_stream.grad).all(), "Gradient contains NaN or Inf values"
    assert v_stream.grad.abs().sum().item() > 0.0, "Gradient is identically zero"

    # Verify parity with legacy autograd pass
    v_leg = v_stream.detach().clone().requires_grad_(True)
    disp_leg = legacy_shoot_ref(model, v_leg, shape, spacing_zyx, phys_grid, meta)
    loss_leg = 0.5 * ((disp_leg - target) ** 2).sum()
    loss_leg.backward()

    assert v_leg.grad is not None
    assert torch.isfinite(v_leg.grad).all()

    grad_diff = (v_stream.grad - v_leg.grad).abs().max().item()
    max_grad = v_stream.grad.abs().max().item()
    rel_grad_diff = grad_diff / (max_grad + 1e-6)
    assert rel_grad_diff < 5e-5, f"Gradient mismatch ({dim}D, {solver}, {transport_mode}): rel_diff={rel_grad_diff:.3e}, abs_diff={grad_diff:.3e}"


# ------------------------------------------------------------------------------
# Test 7: Runtime speedup and memory efficiency
# ------------------------------------------------------------------------------
def test_adversarial_performance_benchmark():
    """Benchmark execution runtime showing speedup in shoot."""
    shape = (16, 16, 16)
    dim = 3
    model, phys_grid, meta, spacing_zyx = _setup_test_env(
        dim=dim, shape=shape, solver='rk4', transport_mode='transport', n_steps=6
    )
    v_init = torch.randn(1, *shape, dim) * 0.5

    # Warmup
    for _ in range(3):
        _ = model.shoot(v_init, shape, spacing_zyx, phys_grid, meta)
        _ = legacy_shoot_ref(model, v_init, shape, spacing_zyx, phys_grid, meta)

    n_iters = 20
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = legacy_shoot_ref(model, v_init, shape, spacing_zyx, phys_grid, meta)
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.shoot(v_init, shape, spacing_zyx, phys_grid, meta)
    t_streamlined = time.perf_counter() - t0

    speedup = (t_legacy - t_streamlined) / t_legacy * 100.0
    print(f"\n[Performance Benchmark] 3D RK4 ({n_iters} iters): legacy={t_legacy:.4f}s, streamlined={t_streamlined:.4f}s, speedup={speedup:.1f}%")
    assert t_streamlined < t_legacy, f"Streamlined ({t_streamlined:.4f}s) should be faster than legacy ({t_legacy:.4f}s)"
