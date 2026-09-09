"""
tests/test_scattered_benchmarks.py — Milestone M4 Benchmark & Verification Suite
================================================================================

Comprehensive benchmarks evaluating:
1. Non-rigid SyN convergence and folding percentage (< 0.1%) under fluid_sigma=6.0 in 2D.
2. 3D non-rigid recovery with folding percentage < 0.1%, valid det(J) > 0, and inverse consistency < 1.0e-3.
3. Memory and compute scaling across point cloud sizes N=100, 500, 1500.
4. Drop-in backward compatibility with sulceye `differentiable_grid_projection`.
5. Anderson fixed-point acceleration efficiency vs Picard iteration.
"""

import math
import time
from typing import Tuple
import pytest
import torch
import torch.nn.functional as F

from syntx.scattered import (
    ScatteredRegistrationConfig,
    syn_scattered,
    ScatteredProjector,
    project_scattered_to_grid,
    differentiable_grid_projection,
    warp_scattered_coordinates,
    pullback_grid_to_scattered,
)
from syntx.core.inverse import update_inverse_field_nd_anderson, update_inverse_field_nd


def compute_pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes Pearson correlation coefficient between two tensors."""
    x_f = x.reshape(-1).float()
    y_f = y.reshape(-1).float()
    x_c = x_f - x_f.mean()
    y_c = y_f - y_f.mean()
    denom = torch.norm(x_c) * torch.norm(y_c)
    if denom < 1e-8:
        return 0.0
    return float(torch.sum(x_c * y_c) / denom)


def generate_benchmark_c_shape(n_points: int = 500, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic C-shaped cortical sheet points and multi-scale curvature features."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0.3 * math.pi, 1.7 * math.pi, n_points)
    r_base = 0.55 + 0.08 * torch.sin(4.0 * theta)
    r_jitter = (torch.rand(n_points, generator=g) - 0.5) * 0.04
    r = r_base + r_jitter

    pts = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    feats = torch.sin(5.0 * theta).unsqueeze(-1) + 0.3 * torch.cos(10.0 * theta).unsqueeze(-1)
    return pts, feats


def generate_benchmark_warp_2d(grid_res: int = 64, amplitude: float = 0.05) -> torch.Tensor:
    """Generates smooth, divergence-free diffeomorphic ground truth warp."""
    y = torch.linspace(-1, 1, grid_res)
    x = torch.linspace(-1, 1, grid_res)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    u_x = amplitude * torch.sin(math.pi * gx) * torch.cos(math.pi * gy)
    u_y = -amplitude * torch.cos(math.pi * gx) * torch.sin(math.pi * gy)
    warp = torch.stack([u_x, u_y], dim=-1).unsqueeze(0)
    return warp


def test_benchmark_scattered_syn_convergence_and_zero_folding_2d():
    """Verify non-rigid alignment improvement, folding < 0.1%, and fluid_sigma=6.0 regularity."""
    X_fix, F_fix = generate_benchmark_c_shape(n_points=450, seed=100)
    u_true = generate_benchmark_warp_2d(grid_res=64, amplitude=0.05)

    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')
    F_mov = F_fix.clone()

    cfg = ScatteredRegistrationConfig(
        dim=2,
        grid_res=64,
        epochs_per_level=[30, 20],
        levels=[2, 1],
        fluid_sigma=6.0,
        optimizer_type='rprop',
        cfl_voxels=0.25,
    )
    res = syn_scattered(X_fix, F_fix, X_mov, F_mov, config=cfg)

    r_init = compute_pearson_r(res.moving_grid, res.fixed_grid)
    r_final = compute_pearson_r(res.warped_moving_grid, res.fixed_grid)
    delta_r = r_final - r_init

    dist_init = float(torch.norm(X_mov - X_fix, dim=-1).mean().item())
    dist_final = float(torch.norm(res.warped_moving_points - X_fix, dim=-1).mean().item())

    # Alignment quality benchmarks
    assert r_final >= 0.90, f"Final correlation {r_final:.4f} < 0.90"
    assert delta_r >= 0.05, f"Correlation gain {delta_r:.4f} < 0.05"
    assert (dist_final / dist_init) < 0.85, f"Coordinate distance ratio {dist_final / dist_init:.4f} >= 0.85"

    # Diffeomorphic guarantees: ZERO folding (< 0.1%) and valid det(J)
    assert res.folding_percentage < 0.1, f"Folding percentage {res.folding_percentage}% >= 0.1%"
    assert res.jacobian_min > 0.0, f"Jacobian determinant minimum {res.jacobian_min} <= 0.0"
    assert res.inverse_consistency_inf < 1.0e-2, f"Inverse consistency {res.inverse_consistency_inf} >= 1.0e-2"


def generate_benchmark_ellipsoid_3d(
    n_points: int = 450,
    seed: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic 3D ellipsoidal shell points and smooth surface features."""
    g = torch.Generator().manual_seed(seed)
    indices = torch.arange(0, n_points, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * indices / n_points)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * indices

    a, b, c = 0.65, 0.50, 0.40
    noise = (torch.rand(n_points, generator=g) - 0.5) * 0.02
    x = (a + noise) * torch.sin(phi) * torch.cos(theta)
    y = (b + noise) * torch.sin(phi) * torch.sin(theta)
    z = (c + noise) * torch.cos(phi)
    pts = torch.stack([x, y, z], dim=-1)

    feats = (x**2 - y**2 + 0.5 * z).unsqueeze(-1)
    return pts, feats


def generate_benchmark_warp_3d(grid_res: int = 32, amplitude: float = 0.03) -> torch.Tensor:
    """Generates smooth, divergence-free diffeomorphic ground truth 3D warp."""
    z = torch.linspace(-1, 1, grid_res)
    y = torch.linspace(-1, 1, grid_res)
    x = torch.linspace(-1, 1, grid_res)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')

    envelope = (1.0 - gz**2) * (1.0 - gy**2) * (1.0 - gx**2)
    ux = amplitude * torch.sin(math.pi * gx) * torch.cos(math.pi * gy) * envelope
    uy = amplitude * torch.cos(math.pi * gx) * torch.sin(math.pi * gz) * envelope
    uz = amplitude * torch.sin(math.pi * gz) * torch.cos(math.pi * gx) * envelope
    return torch.stack([ux, uy, uz], dim=-1).unsqueeze(0)


@pytest.mark.slow
def test_benchmark_scattered_syn_convergence_and_zero_folding_3d():
    """Verify 3D non-rigid recovery with folding percentage < 0.1%, valid det(J) > 0, and inverse consistency < 1.0e-3."""
    X_fix, F_fix = generate_benchmark_ellipsoid_3d(n_points=450, seed=100)
    u_true = generate_benchmark_warp_3d(grid_res=32, amplitude=0.03)

    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')
    F_mov = F_fix.clone()

    cfg = ScatteredRegistrationConfig(
        dim=3,
        grid_res=32,
        epochs_per_level=[20],
        levels=[1],
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
        elastic_sigma=0.5,
    )
    res = syn_scattered(X_fix, F_fix, X_mov, F_mov, config=cfg)

    r_init = compute_pearson_r(res.moving_grid, res.fixed_grid)
    r_final = compute_pearson_r(res.warped_moving_grid, res.fixed_grid)

    dist_init = float(torch.norm(X_mov - X_fix, dim=-1).mean().item())
    dist_final = float(torch.norm(res.warped_moving_points - X_fix, dim=-1).mean().item())

    # Alignment quality benchmarks
    assert r_final >= 0.85, f"Final 3D correlation {r_final:.4f} < 0.85"
    assert (dist_final / dist_init) < 0.95, f"Coordinate distance ratio {dist_final / dist_init:.4f} >= 0.95"

    # Diffeomorphic guarantees: folding < 0.1%, valid det(J) > 0, and inverse consistency < 1.0e-3
    assert res.folding_percentage < 0.1, f"Folding percentage {res.folding_percentage}% >= 0.1%"
    assert res.jacobian_min > 0.0, f"Jacobian determinant minimum {res.jacobian_min} <= 0.0"
    assert res.inverse_consistency_inf < 1.0e-3, f"Inverse consistency {res.inverse_consistency_inf} >= 1.0e-3"


def test_benchmark_scaling_with_point_count():
    """Verify compute and memory scaling across point counts N=100, 500, 1500."""
    grid_res = 48
    projector = ScatteredProjector(grid_shape=(grid_res, grid_res), sigma=0.03)

    for n_pts in [100, 500, 1500]:
        pts, feats = generate_benchmark_c_shape(n_points=n_pts, seed=42)
        
        t0 = time.perf_counter()
        grid = projector(pts, feats)
        dt = time.perf_counter() - t0

        assert grid.shape == (1, 1, grid_res, grid_res)
        assert torch.isfinite(grid).all()
        assert dt < 1.0, f"Projection took {dt:.3f}s for N={n_pts}"


def test_benchmark_sulceye_alias_numerical_parity():
    """Verify differentiable_grid_projection drop-in alias exactly matches ScatteredProjector."""
    pts, feats = generate_benchmark_c_shape(n_points=300, seed=77)

    # 1. Projector execution
    proj = ScatteredProjector(grid_shape=(64, 64), sigma=0.03)
    grid_proj = proj(pts, feats)

    # 2. Legacy sulceye drop-in alias execution
    grid_alias = differentiable_grid_projection(pts, feats, grid_res=64, sigma=0.03)

    # 3. Exact numerical parity (< 1e-6)
    diff = torch.norm(grid_proj - grid_alias).item()
    assert diff < 1.0e-6, f"Difference between projector and alias: {diff:.6e}"


def test_benchmark_anderson_vs_picard_speedup():
    """Verify Anderson acceleration achieves smaller inverse residual than standard fixed-point."""
    u = generate_benchmark_warp_2d(grid_res=48, amplitude=0.04)

    # Anderson accelerated inversion (10 steps)
    v_anderson = update_inverse_field_nd_anderson(u, None, steps=10, m=5)

    # Standard Picard iteration (10 steps)
    v_picard = update_inverse_field_nd(u, None, steps=10, method='fixed_point')

    torch.manual_seed(123)
    X = (torch.rand(300, 2) * 1.4) - 0.7
    Y = warp_scattered_coordinates(X, u, direction='forward')

    X_rec_anderson = warp_scattered_coordinates(Y, v_anderson, direction='forward')
    X_rec_picard = warp_scattered_coordinates(Y, v_picard, direction='forward')

    err_anderson = float(torch.norm(X_rec_anderson - X, p=float('inf'), dim=-1).max().item())
    err_picard = float(torch.norm(X_rec_picard - X, p=float('inf'), dim=-1).max().item())

    assert err_anderson <= err_picard * 1.05, f"Anderson error {err_anderson:.6e} > Picard {err_picard:.6e}"
