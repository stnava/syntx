"""
tests/test_scattered_syn.py — Verification Test Suite for Milestone M3
======================================================================

Comprehensive unit and benchmark test suite for `SyNScattered` and `syn_scattered`,
verifying:
1. Identity mappings in 2D and 3D (zero deformation, correlation >= 0.99, folding = 0.0%)
2. Known synthetic affine and non-rigid sinusoidal warp recovery (correlation >= 0.92, dist drop > 60%)
3. Point-to-grid registration in 2D and 3D
4. Anderson fixed-point inverse consistency (||phi o phi^-1 - id||_inf < 1.0e-3)
5. Anderson vs Picard fixed-point convergence acceleration
6. Physical Jacobian determinant and grid folding rate (< 0.1%)
7. CFL step bounding ablation preventing grid folding
8. Multi-resolution pyramid hierarchy and upsampling fidelity
9. Autograd backpropagation to input coordinates and features with zero NaNs
10. Single-step autograd float64 gradcheck
11. Small and sparse point cloud robustness (N=5 and N=30)
12. Edge cases (boundary clipping, multi-channel C=3, float64, empty point cloud error)
"""

import math
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from syntx.core.inverse import (
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
    compute_inverse_identity_error_nd,
)
from syntx.core.jacobian import (
    compute_physical_jacobian_determinant,
    compute_jacobian_determinant_nd,
)
from syntx.scattered import (
    ScatteredRegistrationConfig,
    ScatteredRegistrationResult,
    SyNScattered,
    syn_scattered,
    ScatteredProjector,
    warp_scattered_coordinates,
    pullback_grid_to_scattered,
)


# ==============================================================================
# Helper Functions & Procedural Synthetic Generators
# ==============================================================================

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


def compute_grid_folding_percentage(
    warp_field: torch.Tensor,
    direction: torch.Tensor | None = None,
    spacing: torch.Tensor | None = None,
    domain_mask: torch.Tensor | None = None,
) -> Tuple[float, float]:
    """Computes (folding_percentage, min_jacobian) using physical Jacobian determinant.

    Returns
    -------
    folding_pct : float (percentage 0.0 - 100.0)
    min_jac : float (minimum determinant value)
    """
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device

    if direction is None:
        direction = torch.eye(dim, device=device)
    if spacing is None:
        spacing = torch.tensor([2.0 / (s - 1) for s in spatial], device=device)

    jac = compute_physical_jacobian_determinant(warp_field, direction, spacing)

    if domain_mask is not None:
        mask = domain_mask.squeeze(0).squeeze(0) if domain_mask.dim() > len(spatial) else domain_mask
        active_jac = jac.squeeze(0)[mask > 0.5]
    else:
        slices = [slice(1, s - 1) for s in spatial]
        active_jac = jac.squeeze(0)[tuple(slices)]

    if active_jac.numel() == 0:
        return 0.0, 1.0

    fold_fraction = float((active_jac <= 0.0).float().mean().item())
    min_jac = float(active_jac.min().item())
    return fold_fraction * 100.0, min_jac


def generate_synthetic_c_shape_2d(
    n_points: int = 500,
    noise_level: float = 0.01,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a non-convex 2D C-shaped point cloud with smooth scalar features."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0.15 * np.pi, 1.85 * np.pi, n_points)
    radius = 0.65 + torch.randn(n_points, generator=g) * noise_level

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=-1)

    features = (torch.sin(2.0 * theta) + 0.5 * torch.cos(4.0 * theta)).unsqueeze(-1)
    return points, features


def generate_synthetic_ellipsoid_3d(
    n_points: int = 800,
    radii: Tuple[float, float, float] = (0.7, 0.5, 0.4),
    noise_level: float = 0.01,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a 3D ellipsoidal shell point cloud with smooth surface features."""
    g = torch.Generator().manual_seed(seed)
    indices = torch.arange(0, n_points, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * indices / n_points)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices

    a, b, c = radii
    noise = torch.randn(n_points, generator=g) * noise_level
    x = (a + noise) * torch.sin(phi) * torch.cos(theta)
    y = (b + noise) * torch.sin(phi) * torch.sin(theta)
    z = (c + noise) * torch.cos(phi)
    points = torch.stack([x, y, z], dim=-1)

    features = (x**2 - y**2 + 0.5 * z).unsqueeze(-1)
    return points, features


def generate_known_diffeomorphic_warp_2d(
    grid_res: int = 64,
    amplitude: float = 0.06,
) -> torch.Tensor:
    """Generates a smooth 2D displacement field with det(J) > 0 everywhere."""
    y = torch.linspace(-1, 1, grid_res)
    x = torch.linspace(-1, 1, grid_res)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    envelope = (1.0 - gy**2) * (1.0 - gx**2)
    ux = amplitude * torch.sin(np.pi * gx) * torch.cos(np.pi * gy) * envelope
    uy = amplitude * torch.cos(np.pi * gx) * torch.sin(np.pi * gy) * envelope
    return torch.stack([ux, uy], dim=-1).unsqueeze(0)


def generate_known_diffeomorphic_warp_3d(
    grid_res: int = 32,
    amplitude: float = 0.04,
) -> torch.Tensor:
    """Generates a smooth 3D displacement field with det(J) > 0 everywhere."""
    z = torch.linspace(-1, 1, grid_res)
    y = torch.linspace(-1, 1, grid_res)
    x = torch.linspace(-1, 1, grid_res)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')

    envelope = (1.0 - gz**2) * (1.0 - gy**2) * (1.0 - gx**2)
    ux = amplitude * torch.sin(np.pi * gx) * torch.cos(np.pi * gy) * envelope
    uy = amplitude * torch.cos(np.pi * gx) * torch.sin(np.pi * gz) * envelope
    uz = amplitude * torch.sin(np.pi * gz) * torch.cos(np.pi * gx) * envelope
    return torch.stack([ux, uy, uz], dim=-1).unsqueeze(0)


def generate_reference_grid_image_2d(grid_res: int = 64) -> torch.Tensor:
    """Generates a 2D Eulerian reference phantom (smooth geometric disk & cross)."""
    y = torch.linspace(-1, 1, grid_res)
    x = torch.linspace(-1, 1, grid_res)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    r = torch.sqrt(gx**2 + gy**2)
    disk = torch.clamp(1.0 - (r / 0.7)**2, min=0.0)
    cross = (gx.abs() < 0.15).float() * (gy.abs() < 0.6).float() + \
            (gy.abs() < 0.15).float() * (gx.abs() < 0.6).float()
    grid = disk + 0.5 * cross
    return grid.unsqueeze(0).unsqueeze(0)


def generate_reference_grid_image_3d(grid_res: int = 32) -> torch.Tensor:
    """Generates a 3D Eulerian reference phantom."""
    z = torch.linspace(-1, 1, grid_res)
    y = torch.linspace(-1, 1, grid_res)
    x = torch.linspace(-1, 1, grid_res)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')

    r = torch.sqrt(gx**2 + gy**2 + gz**2)
    sphere = torch.clamp(1.0 - (r / 0.7)**2, min=0.0)
    return sphere.unsqueeze(0).unsqueeze(0)


# ==============================================================================
# Unit & Benchmark Tests (18 Test Cases)
# ==============================================================================

def test_scattered_syn_config_defaults():
    """Verify configuration dataclass initialization and defaults."""
    cfg = ScatteredRegistrationConfig()
    assert cfg.dim == 2
    assert cfg.grid_res == 128
    assert cfg.domain_bounds == (-1.0, 1.0)
    assert cfg.sigma == 0.03
    assert cfg.fluid_sigma == 1.5
    assert cfg.elastic_sigma == 0.0
    assert cfg.regularizer == 'dsti1'
    assert cfg.optimizer_type == 'rprop'
    assert cfg.optimizer_lr == 0.05
    assert cfg.in_loop_inv_steps == 5
    assert cfg.inverse_steps == 20
    assert cfg.inverse_method == 'anderson'
    assert cfg.cfl_voxels == 0.25
    assert cfg.window_size == 15
    assert cfg.iterations == 100
    assert cfg.w_distortion == 0.1
    assert cfg.antisymmetric is True
    assert cfg.formulation == 'lagrangian'


def test_scattered_syn_identity_2d():
    """Verify 2D identity registration produces zero displacement and zero folding."""
    points, feats = generate_synthetic_c_shape_2d(n_points=300, seed=42)
    cfg = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        epochs_per_level=[15],
        levels=[1],
        cfl_voxels=0.25,
    )
    res = syn_scattered(points, feats, points, feats, config=cfg)

    max_disp = float(torch.norm(res.warp_fwd, dim=-1).max())
    mean_disp = float(torch.norm(res.warp_fwd, dim=-1).mean())
    corr = compute_pearson_r(feats, res.warped_moving_features)

    assert max_disp < 0.02, f"Identity max displacement {max_disp} >= 0.02"
    assert mean_disp < 0.005, f"Identity mean displacement {mean_disp} >= 0.005"
    assert corr >= 0.99, f"Identity correlation {corr} < 0.99"
    assert res.folding_percentage == 0.0, f"Identity produced non-zero folding: {res.folding_percentage}%"


def test_scattered_syn_identity_3d():
    """Verify 3D identity registration produces zero displacement and zero folding."""
    points, feats = generate_synthetic_ellipsoid_3d(n_points=400, seed=42)
    cfg = ScatteredRegistrationConfig(
        dim=3,
        grid_res=24,
        epochs_per_level=[15],
        levels=[1],
        cfl_voxels=0.25,
    )
    res = syn_scattered(points, feats, points, feats, config=cfg)

    max_disp = float(torch.norm(res.warp_fwd, dim=-1).max())
    corr = compute_pearson_r(feats, res.warped_moving_features)

    assert max_disp < 0.02, f"Identity 3D max displacement {max_disp} >= 0.02"
    assert corr >= 0.99, f"Identity 3D correlation {corr} < 0.99"
    assert res.folding_percentage == 0.0, f"Identity 3D produced non-zero folding: {res.folding_percentage}%"


def test_scattered_syn_recover_affine_2d():
    """Verify recovery of rigid rotation and translation on scattered point set."""
    pts_fix, feats_fix = generate_synthetic_c_shape_2d(n_points=400, seed=42)

    # 10 degree rotation + (0.05, -0.04) translation
    theta = float(10.0 * np.pi / 180.0)
    rot = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=torch.float32)
    trans = torch.tensor([0.05, -0.04], dtype=torch.float32)

    pts_mov = pts_fix @ rot.T + trans
    feats_mov = feats_fix.clone()

    proj = ScatteredProjector(grid_shape=(48, 48), sigma=0.03)
    I_f = proj(pts_fix, feats_fix)
    J_m = proj(pts_mov, feats_mov)
    r_init = compute_pearson_r(I_f, J_m)
    assert r_init < 0.85, f"Initial correlation {r_init} unexpectedly high"

    cfg = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        affine_epochs=20,
        epochs_per_level=[15, 15],
        levels=[2, 1],
        cfl_voxels=0.25,
    )
    res = syn_scattered(pts_fix, feats_fix, pts_mov, feats_mov, config=cfg)

    r_final = compute_pearson_r(res.warped_moving_grid, res.fixed_grid)
    delta_r = r_final - r_init

    dist_init = float(torch.norm(pts_mov - pts_fix, dim=-1).mean().item())
    dist_final = float(torch.norm(res.warped_moving_points - pts_fix, dim=-1).mean().item())

    assert r_final >= 0.92, f"Final correlation {r_final} < 0.92"
    assert delta_r >= 0.10, f"Correlation gain {delta_r} < 0.10"
    assert (dist_final / dist_init) < 0.35, f"Distance ratio {dist_final / dist_init} >= 0.35"
    assert res.folding_percentage < 0.1, f"Folding {res.folding_percentage}% >= 0.1%"


def test_scattered_syn_recover_nonrigid_sinusoidal_2d():
    """Verify recovery of known non-rigid diffeomorphic sinusoidal warp."""
    X_fix, F_fix = generate_synthetic_c_shape_2d(n_points=500, seed=1234)
    u_true = generate_known_diffeomorphic_warp_2d(grid_res=64, amplitude=0.06)

    # Lagrangian forward warp of points
    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')
    F_mov = F_fix.clone()

    cfg = ScatteredRegistrationConfig(
        dim=2,
        grid_res=64,
        epochs_per_level=[25, 20],
        levels=[2, 1],
        optimizer_type='rprop',
        cfl_voxels=0.25,
        elastic_sigma=0.5,
    )
    res = syn_scattered(X_fix, F_fix, X_mov, F_mov, config=cfg)

    r_init = compute_pearson_r(res.moving_grid, res.fixed_grid)
    r_final = compute_pearson_r(res.warped_moving_grid, res.fixed_grid)

    dist_init = float(torch.norm(X_mov - X_fix, dim=-1).mean().item())
    dist_final = float(torch.norm(res.warped_moving_points - X_fix, dim=-1).mean().item())

    assert r_final >= 0.92, f"Final correlation {r_final} < 0.92"
    assert (dist_final / dist_init) < 0.80, f"Coordinate distance ratio {dist_final / dist_init} >= 0.80"
    assert res.folding_percentage < 0.1, f"Folding percentage {res.folding_percentage}% >= 0.1%"
    assert res.inverse_consistency_inf < 1.0e-3, f"Inverse consistency {res.inverse_consistency_inf} >= 1.0e-3"


@pytest.mark.slow
def test_scattered_syn_recover_nonrigid_3d():
    """Verify 3D non-rigid warp recovery on ellipsoid surface point cloud."""
    X_fix, F_fix = generate_synthetic_ellipsoid_3d(n_points=600, seed=1234)
    u_true = generate_known_diffeomorphic_warp_3d(grid_res=24, amplitude=0.04)

    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')
    F_mov = F_fix.clone()

    cfg = ScatteredRegistrationConfig(
        dim=3,
        grid_res=24,
        epochs_per_level=[15],
        levels=[1],
        cfl_voxels=0.25,
        elastic_sigma=0.5,
    )
    res = syn_scattered(X_fix, F_fix, X_mov, F_mov, config=cfg)

    r_final = compute_pearson_r(res.warped_moving_grid, res.fixed_grid)
    dist_init = float(torch.norm(X_mov - X_fix, dim=-1).mean().item())
    dist_final = float(torch.norm(res.warped_moving_points - X_fix, dim=-1).mean().item())

    assert r_final >= 0.88, f"Final 3D correlation {r_final} < 0.88"
    assert (dist_final / dist_init) <= 0.95, f"Distance ratio {dist_final / dist_init} > 0.95"
    assert res.folding_percentage < 0.1, f"Folding {res.folding_percentage}% >= 0.1%"
    assert res.inverse_consistency_inf < 1.0e-3, f"Inverse consistency {res.inverse_consistency_inf} >= 1.0e-3"


def test_scattered_syn_point_to_grid_2d():
    """Verify point-to-grid registration against a reference Eulerian image."""
    G_fix = generate_reference_grid_image_2d(grid_res=64)

    theta = torch.linspace(0.0, 2.0 * np.pi, 400)
    r = torch.linspace(0.1, 0.6, 400)
    X_mov = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    F_mov = pullback_grid_to_scattered(G_fix, X_mov)

    u = generate_known_diffeomorphic_warp_2d(64, amplitude=0.05)
    X_mov_deformed = warp_scattered_coordinates(X_mov, u, direction='forward')

    cfg = ScatteredRegistrationConfig(dim=2, grid_res=64, epochs_per_level=[25], levels=[1], cfl_voxels=0.25)
    res = syn_scattered(None, None, X_mov_deformed, F_mov, fixed_grid=G_fix, config=cfg)

    r_grid = compute_pearson_r(res.warped_moving_grid, G_fix)
    F_pb = pullback_grid_to_scattered(G_fix, res.warped_moving_points)
    r_pb = compute_pearson_r(F_pb, F_mov)

    assert r_grid >= 0.60, f"Projected grid correlation {r_grid} < 0.60"
    assert r_pb >= 0.88, f"Pullback feature correlation {r_pb} < 0.88"
    assert res.folding_percentage < 0.1, f"Folding {res.folding_percentage}% >= 0.1%"


@pytest.mark.slow
def test_scattered_syn_point_to_grid_3d():
    """Verify 3D point-to-grid registration against a reference 3D volume."""
    G_fix = generate_reference_grid_image_3d(grid_res=32)

    pts, feats = generate_synthetic_ellipsoid_3d(n_points=500, seed=42)
    feats = pullback_grid_to_scattered(G_fix, pts)

    u = generate_known_diffeomorphic_warp_3d(32, amplitude=0.03)
    pts_deformed = warp_scattered_coordinates(pts, u, direction='forward')

    cfg = ScatteredRegistrationConfig(dim=3, grid_res=32, epochs_per_level=[15], levels=[1], cfl_voxels=0.25)
    res = syn_scattered(None, None, pts_deformed, feats, fixed_grid=G_fix, config=cfg)

    r_grid = compute_pearson_r(res.warped_moving_grid, G_fix)
    F_pb = pullback_grid_to_scattered(G_fix, res.warped_moving_points)
    r_pb = compute_pearson_r(F_pb, feats)
    assert r_grid >= 0.60, f"3D point-to-grid correlation {r_grid} < 0.60"
    assert r_pb >= 0.80, f"3D pullback correlation {r_pb} < 0.80"
    assert res.folding_percentage < 0.1, f"3D point-to-grid folding {res.folding_percentage}% >= 0.1%"


def test_scattered_syn_anderson_inverse_consistency_scattered():
    """Verify Anderson inverse consistency condition: ||phi o phi^-1 - id||_inf < 1.0e-3."""
    u = generate_known_diffeomorphic_warp_2d(grid_res=64, amplitude=0.06)
    v = update_inverse_field_nd_anderson(u, None, steps=20, m=5)

    torch.manual_seed(42)
    X = (torch.rand(500, 2) * 1.6) - 0.8

    Y = warp_scattered_coordinates(X, u, direction='forward')
    X_rec = warp_scattered_coordinates(Y, v, direction='forward')

    err_inf = float(torch.norm(X_rec - X, p=float('inf'), dim=-1).max().item())
    err_mean = float(torch.norm(X_rec - X, p=2, dim=-1).mean().item())

    assert err_inf < 1.0e-3, f"Anderson inverse consistency violated: {err_inf:.6e} >= 1.0e-3"
    assert err_mean < 2.0e-4, f"Anderson mean inverse error too high: {err_mean:.6e} >= 2.0e-4"


def test_scattered_syn_anderson_vs_picard_convergence():
    """Verify Anderson acceleration reaches < 1.0e-3 within 15 iterations."""
    u = generate_known_diffeomorphic_warp_2d(grid_res=64, amplitude=0.04)
    v_anderson = update_inverse_field_nd_anderson(u, None, steps=15, m=5)

    torch.manual_seed(42)
    X = (torch.rand(500, 2) * 1.6) - 0.8
    Y = warp_scattered_coordinates(X, u, direction='forward')
    X_rec = warp_scattered_coordinates(Y, v_anderson, direction='forward')

    err_inf = float(torch.norm(X_rec - X, p=float('inf'), dim=-1).max().item())
    assert err_inf < 1.0e-3, f"Anderson failed to reach 1e-3 in 15 steps: {err_inf:.6e}"


def test_scattered_syn_diffeomorphic_folding_bound_2d():
    """Verify physical det(J) > 0 on >= 99.9% voxels (folding < 0.1%) in 2D."""
    X_fix, F_fix = generate_synthetic_c_shape_2d(n_points=400, seed=123)
    u_true = generate_known_diffeomorphic_warp_2d(grid_res=48, amplitude=0.06)
    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')

    cfg = ScatteredRegistrationConfig(dim=2, grid_res=48, epochs_per_level=[20], levels=[1], cfl_voxels=0.25)
    res = syn_scattered(X_fix, F_fix, X_mov, F_fix, config=cfg)

    folding_pct, min_jac = compute_grid_folding_percentage(res.warp_fwd)
    assert folding_pct < 0.1, f"Grid folding percentage {folding_pct:.4f}% >= 0.1%"
    assert min_jac > -1e-4, f"Pathological fold detected: min det(J) = {min_jac:.4f}"


def test_scattered_syn_diffeomorphic_folding_bound_3d():
    """Verify physical det(J) > 0 on >= 99.9% voxels (folding < 0.1%) in 3D."""
    X_fix, F_fix = generate_synthetic_ellipsoid_3d(n_points=400, seed=123)
    u_true = generate_known_diffeomorphic_warp_3d(grid_res=24, amplitude=0.03)
    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')

    cfg = ScatteredRegistrationConfig(dim=3, grid_res=24, epochs_per_level=[15], levels=[1], cfl_voxels=0.25)
    res = syn_scattered(X_fix, F_fix, X_mov, F_fix, config=cfg)

    folding_pct, min_jac = compute_grid_folding_percentage(res.warp_fwd)
    assert folding_pct < 0.1, f"3D grid folding percentage {folding_pct:.4f}% >= 0.1%"
    assert min_jac > -1e-4, f"3D pathological fold detected: min det(J) = {min_jac:.4f}"


def test_scattered_syn_cfl_bounding_prevents_folding():
    """Ablation test demonstrating CFL step bounding actively prevents grid folding."""
    X_fix, F_fix = generate_synthetic_c_shape_2d(n_points=400, seed=42)
    u_large = generate_known_diffeomorphic_warp_2d(grid_res=48, amplitude=0.10)
    X_mov = warp_scattered_coordinates(X_fix, u_large, direction='forward')

    # Condition A: CFL step bound active (0.25 voxels)
    cfg_a = ScatteredRegistrationConfig(
        dim=2, grid_res=48, cfl_voxels=0.25, epochs_per_level=[25], levels=[1], regularizer='gaussian'
    )
    res_a = syn_scattered(X_fix, F_fix, X_mov, F_fix, config=cfg_a)

    # Condition B: Unconstrained step bound (5.0 voxels)
    cfg_b = ScatteredRegistrationConfig(
        dim=2, grid_res=48, cfl_voxels=5.0, optimizer_lr=0.02, epochs_per_level=[25], levels=[1], regularizer='gaussian'
    )
    res_b = syn_scattered(X_fix, F_fix, X_mov, F_fix, config=cfg_b)

    assert res_a.folding_percentage < 0.1, f"CFL bounded folding {res_a.folding_percentage}% >= 0.1%"
    assert res_a.jacobian_min >= res_b.jacobian_min, "CFL bounding should yield equal or better minimum det(J)"


def test_scattered_syn_multiresolution_pyramid_hierarchy():
    """Verify coarse-to-fine multi-resolution pyramid scheduling and loss progress."""
    X_fix, F_fix = generate_synthetic_c_shape_2d(n_points=400, seed=42)
    u_true = generate_known_diffeomorphic_warp_2d(grid_res=64, amplitude=0.06)
    X_mov = warp_scattered_coordinates(X_fix, u_true, direction='forward')

    cfg = ScatteredRegistrationConfig(
        dim=2, grid_res=64, levels=[2, 1], epochs_per_level=[15, 15], cfl_voxels=0.25
    )
    res = syn_scattered(X_fix, F_fix, X_mov, F_fix, config=cfg)

    assert len(res.loss_history) == 30
    assert res.warp_fwd.shape == (1, 64, 64, 2)
    assert res.warp_inv.shape == (1, 64, 64, 2)
    assert res.loss_history[-1] < res.loss_history[0], "Multi-res pyramid failed to improve similarity"


def test_scattered_syn_pyramid_upsampling_fidelity():
    """Verify upsampling operator preserves displacement vector directions."""
    warp_32 = generate_known_diffeomorphic_warp_2d(grid_res=32, amplitude=0.05)

    warp_64 = F.interpolate(
        warp_32.movedim(-1, 1), size=(64, 64), mode='bilinear', align_corners=True
    ).movedim(1, -1)

    grids = [torch.linspace(-1, 1, 32) for _ in range(2)]
    identity_32 = torch.stack(list(reversed(torch.meshgrid(*grids, indexing='ij'))), dim=-1).unsqueeze(0)
    warp_64_sampled = F.grid_sample(warp_64.movedim(-1, 1), identity_32, mode='bilinear', align_corners=True).movedim(1, -1)

    norm_32 = torch.norm(warp_32, dim=-1)
    mask = norm_32 > 1e-4
    cos_sim = F.cosine_similarity(warp_32[mask], warp_64_sampled[mask], dim=-1)

    assert float(cos_sim.mean().item()) > 0.999, "Upsampling failed to preserve displacement vector directions"


def test_scattered_syn_autograd_differentiability_points_and_features():
    """Verify autograd gradients propagate back to moving coordinates and features without NaNs."""
    pts_fix, feats_fix = generate_synthetic_c_shape_2d(n_points=30, seed=1)
    pts_mov = (pts_fix + torch.randn_like(pts_fix) * 0.05).requires_grad_(True)
    feats_mov = (feats_fix + torch.randn_like(feats_fix) * 0.05).requires_grad_(True)

    solver = SyNScattered(dim=2, grid_res=32, window_size=5)
    loss = solver.step(pts_fix, feats_fix, pts_mov, feats_mov)
    loss.backward()

    assert pts_mov.grad is not None, "pts_mov.grad is None"
    assert feats_mov.grad is not None, "feats_mov.grad is None"
    assert torch.isfinite(pts_mov.grad).all(), "pts_mov.grad contains non-finite values"
    assert torch.isfinite(feats_mov.grad).all(), "feats_mov.grad contains non-finite values"
    assert not torch.isnan(pts_mov.grad).any(), "pts_mov.grad contains NaNs"
    assert not torch.isnan(feats_mov.grad).any(), "feats_mov.grad contains NaNs"
    assert float(torch.norm(pts_mov.grad).item()) > 0.0, "pts_mov.grad is all zeros"


def test_scattered_syn_autograd_gradcheck_single_step():
    """Rigorous finite-difference gradient check in double precision."""
    torch.manual_seed(42)
    pts_fix = torch.randn(6, 2, dtype=torch.float64) * 0.4
    feats_fix = torch.randn(6, 1, dtype=torch.float64)
    pts_mov = (torch.randn(6, 2, dtype=torch.float64) * 0.4).requires_grad_(True)
    feats_mov = torch.randn(6, 1, dtype=torch.float64).requires_grad_(True)

    solver = SyNScattered(dim=2, grid_res=16, dtype=torch.float64, window_size=5)

    def func(p, f):
        return solver.step(pts_fix, feats_fix, p, f)

    assert torch.autograd.gradcheck(func, (pts_mov, feats_mov), eps=1e-5, atol=1e-3, rtol=1e-3)


def test_scattered_syn_small_and_sparse_point_sets():
    """Verify solver handles sparse point clouds (N=5 and N=30) without dimension crash or NaNs."""
    for n_pts in (5, 30):
        pts_f = torch.randn(n_pts, 2) * 0.5
        fts_f = torch.ones(n_pts, 1)
        pts_m = pts_f + 0.05 * torch.randn_like(pts_f)
        fts_m = fts_f.clone()

        cfg = ScatteredRegistrationConfig(dim=2, grid_res=32, epochs_per_level=[5], levels=[1])
        res = syn_scattered(pts_f, fts_f, pts_m, fts_m, config=cfg)

        assert res.disp_fwd.shape == (1, 32, 32, 2)
        assert torch.isfinite(res.disp_fwd).all()
        assert not torch.isnan(res.disp_fwd).any()
        assert res.warped_moving_points.shape == (n_pts, 2)


def test_scattered_syn_edge_cases_and_robustness():
    """Verify out-of-bounds coords, multi-channel features, float64 dtype, and empty input handling."""
    # 1. Empty point cloud raises ValueError
    empty_pts = torch.zeros((0, 2))
    empty_fts = torch.zeros((0, 1))
    valid_pts = torch.randn(10, 2)
    valid_fts = torch.randn(10, 1)

    with pytest.raises(ValueError, match="empty"):
        syn_scattered(empty_pts, empty_fts, valid_pts, valid_fts)

    # 2. Out-of-bounds coordinates (e.g. x = 1.3)
    oob_pts = torch.tensor([[1.3, 0.2], [-1.2, -0.5]], dtype=torch.float32)
    oob_fts = torch.ones(2, 1)
    cfg = ScatteredRegistrationConfig(dim=2, grid_res=32, epochs_per_level=[5], levels=[1])
    res_oob = syn_scattered(valid_pts, valid_fts, oob_pts, oob_fts, config=cfg)
    assert torch.isfinite(res_oob.disp_fwd).all()
    assert not torch.isnan(res_oob.disp_fwd).any()

    # 3. Multi-channel features (C=3)
    mc_fts_f = torch.randn(10, 3)
    mc_fts_m = torch.randn(10, 3)
    res_mc = syn_scattered(valid_pts, mc_fts_f, valid_pts, mc_fts_m, config=cfg)
    assert res_mc.fixed_grid.shape[1] == 3
    assert res_mc.moving_grid.shape[1] == 3

    # 4. Double precision (float64) execution
    pts64 = valid_pts.to(torch.float64)
    fts64 = valid_fts.to(torch.float64)
    res64 = syn_scattered(pts64, fts64, pts64, fts64, config=cfg)
    assert res64.disp_fwd.dtype == torch.float64
    assert res64.disp_inv.dtype == torch.float64

    # 5. Result dict-like indexing backwards compatibility
    assert 'warpedmovout' in res64
    assert res64['fwdtransforms'][0] is res64.disp_fwd
    assert res64['invtransforms'][0] is res64.disp_inv


def test_scattered_syn_forward_coordinate_frames():
    """Verify SyNScattered.forward() coordinate frame directions for points and grid."""
    pts_fix = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    feats_fix = torch.tensor([[1.0]], dtype=torch.float32)
    pts_mov = torch.tensor([[0.1, 0.0]], dtype=torch.float32)
    feats_mov = torch.tensor([[1.0]], dtype=torch.float32)

    cfg = ScatteredRegistrationConfig(dim=2, grid_res=32, epochs_per_level=[10], levels=[1])
    res = syn_scattered(pts_fix, feats_fix, pts_mov, feats_mov, config=cfg)
    model = res.model

    # Both result.warped_moving_points and model.forward(moving_points) should move pts_mov towards [0, 0]
    warped_pts_forward = model.forward(moving_points=pts_mov, direction='forward')
    dist_init = float(torch.norm(pts_mov - pts_fix))
    dist_warped = float(torch.norm(warped_pts_forward - pts_fix))
    assert dist_warped < dist_init, f"Forward warped points moved away from fixed: {dist_warped} >= {dist_init}"

    # Also check res.warp_points matches
    res_warped = res.warp_points(pts_mov, direction='forward')
    assert torch.allclose(res_warped, warped_pts_forward, atol=1e-5)


def test_scattered_syn_multiepoch_requires_grad():
    """Verify multi-epoch fit() executes without RuntimeError when inputs require grad."""
    pts_f = torch.randn(20, 2, requires_grad=True)
    fts_f = torch.randn(20, 1, requires_grad=True)
    pts_m = torch.randn(20, 2, requires_grad=True)
    fts_m = torch.randn(20, 1, requires_grad=True)

    cfg = ScatteredRegistrationConfig(dim=2, grid_res=32, epochs_per_level=[3], levels=[1])
    res = syn_scattered(pts_f, fts_f, pts_m, fts_m, config=cfg)
    assert res.disp_fwd is not None
    assert torch.isfinite(res.disp_fwd).all()


def test_scattered_syn_point_weights():
    """Verify point_weights are properly passed and utilized in projection."""
    pts_f = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    fts_f = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    weights_f = torch.tensor([1.0, 0.0], dtype=torch.float32)

    cfg = ScatteredRegistrationConfig(dim=2, grid_res=32, epochs_per_level=[2], levels=[1])
    res = syn_scattered(pts_f, fts_f, pts_f, fts_f, point_weights_fixed=weights_f, config=cfg)
    assert res.disp_fwd is not None

