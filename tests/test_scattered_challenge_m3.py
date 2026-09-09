"""
tests/test_scattered_challenge_m3.py — Empirical Challenge Suite for Milestone M3
================================================================================

Independent empirical stress-test of Milestone M3 core mathematical guarantees:
1. Symmetric Inverse Consistency: ||phi o phi^-1 - id||_inf < 10^-3 on scattered points.
2. Diffeomorphic Folding Rate: Grid folding percentage < 0.1% (det(J) <= 0 fraction < 0.001).
3. CFL Sensitivity Ablation: cfl_voxels=0.25 vs cfl_voxels=1.0 proving CFL bounding actively prevents folding.

Evaluates across:
- Non-trivial 2D Crescent point cloud (non-convex, varying thickness)
- Non-trivial 2D Multi-turn Archimedean Spiral point cloud (complex winding topology)
- Non-trivial 3D Ellipsoidal Shell point cloud (embedded curved 2-manifold in R^3)
"""

import math
from typing import Tuple, Dict, Any, List
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from syntx.core.inverse import (
    update_inverse_field_nd_anderson,
    compute_inverse_identity_error_nd,
)
from syntx.core.jacobian import compute_physical_jacobian_determinant
from syntx.scattered import (
    ScatteredRegistrationConfig,
    ScatteredRegistrationResult,
    SyNScattered,
    syn_scattered,
    warp_scattered_coordinates,
    pullback_grid_to_scattered,
)


# ==============================================================================
# Helper & Metric Utilities
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


def evaluate_exact_jacobian_metrics(
    warp_field: torch.Tensor,
    domain_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Computes exact physical Jacobian determinant statistics.

    Parameters
    ----------
    warp_field : torch.Tensor of shape (1, *spatial, d)
    domain_mask : optional mask

    Returns
    -------
    dict with:
      - 'folding_percentage': % of voxels with det(J) <= 0
      - 'min_jacobian': minimum det(J)
      - 'mean_jacobian': mean det(J)
      - 'std_jacobian': std of det(J)
      - 'folded_voxel_count': number of folded voxels
      - 'total_voxel_count': total evaluated voxels
    """
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device
    dtype = warp_field.dtype

    direction = torch.eye(dim, device=device, dtype=dtype)
    spacing = torch.tensor([2.0 / (s - 1) for s in spatial], device=device, dtype=dtype)

    jac = compute_physical_jacobian_determinant(warp_field, direction=direction, spacing=spacing)

    if domain_mask is not None:
        mask = domain_mask.squeeze(0).squeeze(0) if domain_mask.dim() > len(spatial) else domain_mask
        active_jac = jac.squeeze(0)[mask > 0.5]
    else:
        # Exclude outermost 1-voxel boundary where central differences use replicate padding
        slices = [slice(1, s - 1) for s in spatial]
        active_jac = jac.squeeze(0)[tuple(slices)]

    total_voxels = active_jac.numel()
    folded_voxels = int((active_jac <= 0.0).sum().item())
    folding_pct = (folded_voxels / total_voxels) * 100.0 if total_voxels > 0 else 0.0

    return {
        'folding_percentage': folding_pct,
        'min_jacobian': float(active_jac.min().item()),
        'mean_jacobian': float(active_jac.mean().item()),
        'std_jacobian': float(active_jac.std().item()),
        'folded_voxel_count': folded_voxels,
        'total_voxel_count': total_voxels,
    }


def evaluate_scattered_inverse_consistency(
    points: torch.Tensor,
    warp_fwd: torch.Tensor,
    warp_inv: torch.Tensor,
    domain_bounds: Tuple[float, float] = (-1.0, 1.0),
    coord_convention: str = 'xyz',
) -> Dict[str, float]:
    """Measures exact maximum and mean inverse coordinate error ||phi o phi^-1 - id||.

    Maps points forward then backward: X -> Y = X + u(X) -> X_rec = Y + v(Y).
    """
    Y = warp_scattered_coordinates(
        coords=points,
        displacement_field=warp_fwd,
        direction='forward',
        domain_bounds=domain_bounds,
        coord_convention=coord_convention,
    )
    X_rec = warp_scattered_coordinates(
        coords=Y,
        displacement_field=warp_inv,
        direction='forward',
        domain_bounds=domain_bounds,
        coord_convention=coord_convention,
    )

    diff = X_rec - points
    # Coordinate-wise max norm: max_i max_d |X_rec[i, d] - X[i, d]|
    max_err = float(torch.abs(diff).max().item())
    mean_err = float(torch.norm(diff, p=2, dim=-1).mean().item())
    p99_err = float(torch.quantile(torch.norm(diff, p=2, dim=-1), 0.99).item())

    return {
        'max_inverse_error': max_err,
        'mean_inverse_error': mean_err,
        'p99_inverse_error': p99_err,
    }


# ==============================================================================
# Procedural Geometry & Deformation Generators
# ==============================================================================

def generate_crescent_2d(
    n_points: int = 500,
    noise_level: float = 0.008,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates non-convex 2D Crescent point cloud with spatially varying features.

    Shape defined between outer arc r1(theta) and inner arc r2(theta).
    """
    g = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0.18 * np.pi, 1.82 * np.pi, n_points)
    # Inner and outer radius varies with theta creating crescent taper
    t = (theta - np.pi) / (0.82 * np.pi)  # in [-1, 1]
    thickness = 0.22 * (1.0 - t**2) + 0.04
    center_r = 0.60 + 0.05 * torch.cos(theta)

    radial_offset = (torch.rand(n_points, generator=g) - 0.5) * thickness
    radius = center_r + radial_offset + torch.randn(n_points, generator=g) * noise_level

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=-1)

    # Multi-frequency smooth scalar feature
    features = (torch.sin(3.0 * theta) + 0.6 * torch.cos(5.0 * theta) + 0.4 * torch.sin(2.0 * np.pi * radius)).unsqueeze(-1)
    return points, features


def generate_spiral_2d(
    n_points: int = 600,
    noise_level: float = 0.005,
    seed: int = 101,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates 2D Archimedean multi-turn spiral with challenging winding topology.

    Adjacent turns are close in Euclidean space but distant along the geodesic.
    """
    g = torch.Generator().manual_seed(seed)
    # 2.2 turns from 1.0 pi to 5.4 pi
    theta = torch.linspace(1.0 * np.pi, 5.4 * np.pi, n_points)
    # Archimedean radius r = a + b * theta mapped to [0.15, 0.72]
    radius = 0.12 + 0.035 * theta + torch.randn(n_points, generator=g) * noise_level

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=-1)

    # Feature encodes position along the spiral
    features = (torch.cos(1.5 * theta) + 0.5 * torch.sin(4.0 * theta)).unsqueeze(-1)
    return points, features


def generate_ellipsoid_shell_3d(
    n_points: int = 800,
    radii: Tuple[float, float, float] = (0.72, 0.52, 0.36),
    noise_level: float = 0.008,
    seed: int = 202,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates 3D tri-axial ellipsoidal shell (curved 2-manifold in R^3)."""
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

    # Smooth spherical harmonic-like feature
    features = (x**2 - y**2 + 1.2 * z + 0.4 * x * z).unsqueeze(-1)
    return points, features


def apply_synthetic_deformation_2d(
    points: torch.Tensor,
    amplitude: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies smooth non-rigid multi-frequency sinusoidal displacement to 2D points.

    Returns (deformed_points, ground_truth_displacement_at_points).
    """
    x = points[:, 0]
    y = points[:, 1]
    envelope = torch.clamp((1.0 - x**2) * (1.0 - y**2), min=0.0)

    ux = amplitude * (torch.sin(np.pi * x) * torch.cos(np.pi * y) + 0.3 * torch.sin(2 * np.pi * y)) * envelope
    uy = amplitude * (torch.cos(np.pi * x) * torch.sin(np.pi * y) + 0.3 * torch.cos(2 * np.pi * x)) * envelope
    disp = torch.stack([ux, uy], dim=-1)
    return points + disp, disp


def apply_synthetic_deformation_3d(
    points: torch.Tensor,
    amplitude: float = 0.045,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies smooth non-rigid 3D sinusoidal displacement to 3D points."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    envelope = torch.clamp((1.0 - x**2) * (1.0 - y**2) * (1.0 - z**2), min=0.0)

    ux = amplitude * torch.sin(np.pi * x) * torch.cos(np.pi * y) * envelope
    uy = amplitude * torch.cos(np.pi * x) * torch.sin(np.pi * z) * envelope
    uz = amplitude * torch.sin(np.pi * z) * torch.cos(np.pi * x) * envelope
    disp = torch.stack([ux, uy, uz], dim=-1)
    return points + disp, disp


# ==============================================================================
# Challenge Execution Runner
# ==============================================================================

def run_empirical_challenge_suite() -> Dict[str, Any]:
    """Executes the complete empirical stress test suite across all geometries."""
    results = {}

    print("================================================================================")
    print("M3 CHALLENGE 1: 2D Crescent Geometry (Non-convex, varying thickness)")
    print("================================================================================")
    pts_fix, feats_fix = generate_crescent_2d(n_points=500, seed=42)
    pts_mov, u_true = apply_synthetic_deformation_2d(pts_fix, amplitude=0.07)
    feats_mov = feats_fix.clone()

    cfg_crescent = ScatteredRegistrationConfig(
        dim=2,
        grid_res=64,
        epochs_per_level=[30, 20],
        levels=[2, 1],
        optimizer_type='rprop',
        regularizer='dsti1',
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
        elastic_sigma=0.5,
    )
    res_crescent = syn_scattered(pts_fix, feats_fix, pts_mov, feats_mov, config=cfg_crescent)

    inv_crescent = evaluate_scattered_inverse_consistency(pts_fix, res_crescent.warp_fwd, res_crescent.warp_inv)
    jac_crescent = evaluate_exact_jacobian_metrics(res_crescent.warp_fwd)

    dist_init_crescent = float(torch.norm(pts_mov - pts_fix, dim=-1).mean().item())
    dist_final_crescent = float(torch.norm(res_crescent.warped_moving_points - pts_fix, dim=-1).mean().item())
    r_final_crescent = compute_pearson_r(res_crescent.warped_moving_grid, res_crescent.fixed_grid)

    results['crescent'] = {
        'inv': inv_crescent,
        'jac': jac_crescent,
        'dist_init': dist_init_crescent,
        'dist_final': dist_final_crescent,
        'dist_reduction': (1.0 - dist_final_crescent / dist_init_crescent) * 100.0,
        'r_final': r_final_crescent,
    }
    print(f"  Max Inverse Coordinate Error : {inv_crescent['max_inverse_error']:.6e} (Criterion: < 1.0e-3)")
    print(f"  Mean Inverse Coordinate Error: {inv_crescent['mean_inverse_error']:.6e}")
    print(f"  Grid Folding Percentage      : {jac_crescent['folding_percentage']:.4f}% (Criterion: < 0.1%)")
    print(f"  Minimum det(J)               : {jac_crescent['min_jacobian']:.4f}")
    print(f"  Distance Reduction           : {results['crescent']['dist_reduction']:.2f}% ({dist_init_crescent:.4f} -> {dist_final_crescent:.4f})")
    print(f"  Final Pearson r              : {r_final_crescent:.4f}")

    print("\n================================================================================")
    print("M3 CHALLENGE 2: 2D Spiral Geometry (Multi-turn Archimedean Spiral)")
    print("================================================================================")
    pts_fix_sp, feats_fix_sp = generate_spiral_2d(n_points=600, seed=101)
    pts_mov_sp, u_true_sp = apply_synthetic_deformation_2d(pts_fix_sp, amplitude=0.06)
    feats_mov_sp = feats_fix_sp.clone()

    cfg_spiral = ScatteredRegistrationConfig(
        dim=2,
        grid_res=64,
        epochs_per_level=[30, 20],
        levels=[2, 1],
        optimizer_type='rprop',
        regularizer='dsti1',
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
        elastic_sigma=0.5,
    )
    res_spiral = syn_scattered(pts_fix_sp, feats_fix_sp, pts_mov_sp, feats_mov_sp, config=cfg_spiral)

    inv_spiral = evaluate_scattered_inverse_consistency(pts_fix_sp, res_spiral.warp_fwd, res_spiral.warp_inv)
    jac_spiral = evaluate_exact_jacobian_metrics(res_spiral.warp_fwd)

    dist_init_spiral = float(torch.norm(pts_mov_sp - pts_fix_sp, dim=-1).mean().item())
    dist_final_spiral = float(torch.norm(res_spiral.warped_moving_points - pts_fix_sp, dim=-1).mean().item())
    r_final_spiral = compute_pearson_r(res_spiral.warped_moving_grid, res_spiral.fixed_grid)

    results['spiral'] = {
        'inv': inv_spiral,
        'jac': jac_spiral,
        'dist_init': dist_init_spiral,
        'dist_final': dist_final_spiral,
        'dist_reduction': (1.0 - dist_final_spiral / dist_init_spiral) * 100.0,
        'r_final': r_final_spiral,
    }
    print(f"  Max Inverse Coordinate Error : {inv_spiral['max_inverse_error']:.6e} (Criterion: < 1.0e-3)")
    print(f"  Mean Inverse Coordinate Error: {inv_spiral['mean_inverse_error']:.6e}")
    print(f"  Grid Folding Percentage      : {jac_spiral['folding_percentage']:.4f}% (Criterion: < 0.1%)")
    print(f"  Minimum det(J)               : {jac_spiral['min_jacobian']:.4f}")
    print(f"  Distance Reduction           : {results['spiral']['dist_reduction']:.2f}% ({dist_init_spiral:.4f} -> {dist_final_spiral:.4f})")
    print(f"  Final Pearson r              : {r_final_spiral:.4f}")

    print("\n================================================================================")
    print("M3 CHALLENGE 3: 3D Ellipsoid Shell Geometry (Curved 2-manifold in R^3)")
    print("================================================================================")
    pts_fix_3d, feats_fix_3d = generate_ellipsoid_shell_3d(n_points=700, seed=202)
    pts_mov_3d, u_true_3d = apply_synthetic_deformation_3d(pts_fix_3d, amplitude=0.04)
    feats_mov_3d = feats_fix_3d.clone()

    cfg_3d = ScatteredRegistrationConfig(
        dim=3,
        grid_res=32,
        epochs_per_level=[20],
        levels=[1],
        optimizer_type='rprop',
        regularizer='dsti1',
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
        elastic_sigma=0.5,
    )
    res_3d = syn_scattered(pts_fix_3d, feats_fix_3d, pts_mov_3d, feats_mov_3d, config=cfg_3d)

    inv_3d = evaluate_scattered_inverse_consistency(pts_fix_3d, res_3d.warp_fwd, res_3d.warp_inv)
    jac_3d = evaluate_exact_jacobian_metrics(res_3d.warp_fwd)

    dist_init_3d = float(torch.norm(pts_mov_3d - pts_fix_3d, dim=-1).mean().item())
    dist_final_3d = float(torch.norm(res_3d.warped_moving_points - pts_fix_3d, dim=-1).mean().item())
    r_final_3d = compute_pearson_r(res_3d.warped_moving_grid, res_3d.fixed_grid)

    results['ellipsoid_3d'] = {
        'inv': inv_3d,
        'jac': jac_3d,
        'dist_init': dist_init_3d,
        'dist_final': dist_final_3d,
        'dist_reduction': (1.0 - dist_final_3d / dist_init_3d) * 100.0,
        'r_final': r_final_3d,
    }
    print(f"  Max Inverse Coordinate Error : {inv_3d['max_inverse_error']:.6e} (Criterion: < 1.0e-3)")
    print(f"  Mean Inverse Coordinate Error: {inv_3d['mean_inverse_error']:.6e}")
    print(f"  Grid Folding Percentage      : {jac_3d['folding_percentage']:.4f}% (Criterion: < 0.1%)")
    print(f"  Minimum det(J)               : {jac_3d['min_jacobian']:.4f}")
    print(f"  Distance Reduction           : {results['ellipsoid_3d']['dist_reduction']:.2f}% ({dist_init_3d:.4f} -> {dist_final_3d:.4f})")
    print(f"  Final Pearson r              : {r_final_3d:.4f}")

    print("\n================================================================================")
    print("M3 CHALLENGE 4: CFL Sensitivity Ablation (cfl_voxels=0.25 vs cfl_voxels=1.0)")
    print("================================================================================")
    # Under an aggressive large-amplitude deformation and learning rate:
    pts_cfl_fix, feats_cfl_fix = generate_crescent_2d(n_points=400, seed=999)
    pts_cfl_mov, _ = apply_synthetic_deformation_2d(pts_cfl_fix, amplitude=0.12)  # aggressive deformation
    feats_cfl_mov = feats_cfl_fix.clone()

    # Condition 1: CFL Bounded (cfl_voxels=0.25)
    cfg_cfl_025 = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        epochs_per_level=[30],
        levels=[1],
        optimizer_type='rprop',
        optimizer_lr=0.10,
        regularizer='gaussian',
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
    )
    res_cfl_025 = syn_scattered(pts_cfl_fix, feats_cfl_fix, pts_cfl_mov, feats_cfl_mov, config=cfg_cfl_025)
    jac_cfl_025 = evaluate_exact_jacobian_metrics(res_cfl_025.warp_fwd)
    inv_cfl_025 = evaluate_scattered_inverse_consistency(pts_cfl_fix, res_cfl_025.warp_fwd, res_cfl_025.warp_inv)

    # Condition 2: Relaxed CFL Bound (cfl_voxels=1.0)
    cfg_cfl_100 = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        epochs_per_level=[30],
        levels=[1],
        optimizer_type='rprop',
        optimizer_lr=0.10,
        regularizer='gaussian',
        cfl_voxels=1.0,
        in_loop_inv_steps=5,
        inverse_steps=20,
    )
    res_cfl_100 = syn_scattered(pts_cfl_fix, feats_cfl_fix, pts_cfl_mov, feats_cfl_mov, config=cfg_cfl_100)
    jac_cfl_100 = evaluate_exact_jacobian_metrics(res_cfl_100.warp_fwd)
    inv_cfl_100 = evaluate_scattered_inverse_consistency(pts_cfl_fix, res_cfl_100.warp_fwd, res_cfl_100.warp_inv)

    # Condition 3: Highly Relaxed CFL Bound (cfl_voxels=3.0) for boundary stress
    cfg_cfl_300 = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        epochs_per_level=[30],
        levels=[1],
        optimizer_type='rprop',
        optimizer_lr=0.15,
        regularizer='gaussian',
        cfl_voxels=3.0,
        in_loop_inv_steps=5,
        inverse_steps=20,
    )
    res_cfl_300 = syn_scattered(pts_cfl_fix, feats_cfl_fix, pts_cfl_mov, feats_cfl_mov, config=cfg_cfl_300)
    jac_cfl_300 = evaluate_exact_jacobian_metrics(res_cfl_300.warp_fwd)
    inv_cfl_300 = evaluate_scattered_inverse_consistency(pts_cfl_fix, res_cfl_300.warp_fwd, res_cfl_300.warp_inv)

    results['cfl_ablation'] = {
        'cfl_025': {'jac': jac_cfl_025, 'inv': inv_cfl_025},
        'cfl_100': {'jac': jac_cfl_100, 'inv': inv_cfl_100},
        'cfl_300': {'jac': jac_cfl_300, 'inv': inv_cfl_300},
    }

    print(f"  CFL = 0.25 voxels: Folding = {jac_cfl_025['folding_percentage']:.4f}%, Min det(J) = {jac_cfl_025['min_jacobian']:.4f}, Max Inv Err = {inv_cfl_025['max_inverse_error']:.6e}")
    print(f"  CFL = 1.00 voxels: Folding = {jac_cfl_100['folding_percentage']:.4f}%, Min det(J) = {jac_cfl_100['min_jacobian']:.4f}, Max Inv Err = {inv_cfl_100['max_inverse_error']:.6e}")
    print(f"  CFL = 3.00 voxels: Folding = {jac_cfl_300['folding_percentage']:.4f}%, Min det(J) = {jac_cfl_300['min_jacobian']:.4f}, Max Inv Err = {inv_cfl_300['max_inverse_error']:.6e}")

    return results


# ==============================================================================
# Pytest Test Cases Asserting Mathematical Guarantees
# ==============================================================================

def test_m3_challenge_crescent_2d_guarantees():
    """Verify 2D Crescent satisfies both inverse consistency < 1e-3 and folding < 0.1%."""
    pts_fix, feats_fix = generate_crescent_2d(n_points=500, seed=42)
    pts_mov, _ = apply_synthetic_deformation_2d(pts_fix, amplitude=0.07)

    cfg = ScatteredRegistrationConfig(
        dim=2,
        grid_res=64,
        epochs_per_level=[30, 20],
        levels=[2, 1],
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
        elastic_sigma=0.5,
    )
    res = syn_scattered(pts_fix, feats_fix, pts_mov, feats_fix, config=cfg)

    inv = evaluate_scattered_inverse_consistency(pts_fix, res.warp_fwd, res.warp_inv)
    jac = evaluate_exact_jacobian_metrics(res.warp_fwd)

    assert inv['max_inverse_error'] < 1.0e-3, (
        f"2D Crescent violated inverse consistency: {inv['max_inverse_error']:.6e} >= 1.0e-3"
    )
    assert jac['folding_percentage'] < 0.1, (
        f"2D Crescent violated folding rate: {jac['folding_percentage']:.4f}% >= 0.1%"
    )
    assert jac['min_jacobian'] > -1e-4, f"Severe negative folding: min det(J) = {jac['min_jacobian']}"


def test_m3_challenge_spiral_2d_guarantees():
    """Verify 2D Spiral satisfies both inverse consistency < 1e-3 and folding < 0.1%."""
    pts_fix, feats_fix = generate_spiral_2d(n_points=600, seed=101)
    pts_mov, _ = apply_synthetic_deformation_2d(pts_fix, amplitude=0.06)

    cfg = ScatteredRegistrationConfig(
        dim=2,
        grid_res=64,
        epochs_per_level=[30, 20],
        levels=[2, 1],
        cfl_voxels=0.25,
        in_loop_inv_steps=5,
        inverse_steps=20,
        elastic_sigma=0.5,
    )
    res = syn_scattered(pts_fix, feats_fix, pts_mov, feats_fix, config=cfg)

    inv = evaluate_scattered_inverse_consistency(pts_fix, res.warp_fwd, res.warp_inv)
    jac = evaluate_exact_jacobian_metrics(res.warp_fwd)

    assert inv['max_inverse_error'] < 1.0e-3, (
        f"2D Spiral violated inverse consistency: {inv['max_inverse_error']:.6e} >= 1.0e-3"
    )
    assert jac['folding_percentage'] < 0.1, (
        f"2D Spiral violated folding rate: {jac['folding_percentage']:.4f}% >= 0.1%"
    )
    assert jac['min_jacobian'] > -1e-4, f"Severe negative folding: min det(J) = {jac['min_jacobian']}"


@pytest.mark.slow
def test_m3_challenge_ellipsoid_shell_3d_guarantees():
    """Verify 3D Ellipsoid shell satisfies both inverse consistency < 1e-3 and folding < 0.1%."""
    pts_fix, feats_fix = generate_ellipsoid_shell_3d(n_points=700, seed=202)
    pts_mov, _ = apply_synthetic_deformation_3d(pts_fix, amplitude=0.04)

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
    res = syn_scattered(pts_fix, feats_fix, pts_mov, feats_fix, config=cfg)

    inv = evaluate_scattered_inverse_consistency(pts_fix, res.warp_fwd, res.warp_inv)
    jac = evaluate_exact_jacobian_metrics(res.warp_fwd)

    assert inv['max_inverse_error'] < 1.0e-3, (
        f"3D Ellipsoid shell violated inverse consistency: {inv['max_inverse_error']:.6e} >= 1.0e-3"
    )
    assert jac['folding_percentage'] < 0.1, (
        f"3D Ellipsoid shell violated folding rate: {jac['folding_percentage']:.4f}% >= 0.1%"
    )
    assert jac['min_jacobian'] > -1e-4, f"Severe negative folding: min det(J) = {jac['min_jacobian']}"


def test_m3_challenge_cfl_sensitivity_comparison():
    """Verify that CFL bounding (0.25 vs 1.0) preserves diffeomorphic regularity."""
    pts_fix, feats_fix = generate_crescent_2d(n_points=400, seed=999)
    pts_mov, _ = apply_synthetic_deformation_2d(pts_fix, amplitude=0.12)

    cfg_025 = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        epochs_per_level=[25],
        levels=[1],
        optimizer_type='rprop',
        optimizer_lr=0.10,
        regularizer='gaussian',
        cfl_voxels=0.25,
    )
    res_025 = syn_scattered(pts_fix, feats_fix, pts_mov, feats_fix, config=cfg_025)
    jac_025 = evaluate_exact_jacobian_metrics(res_025.warp_fwd)

    cfg_100 = ScatteredRegistrationConfig(
        dim=2,
        grid_res=48,
        epochs_per_level=[25],
        levels=[1],
        optimizer_type='rprop',
        optimizer_lr=0.10,
        regularizer='gaussian',
        cfl_voxels=1.0,
    )
    res_100 = syn_scattered(pts_fix, feats_fix, pts_mov, feats_fix, config=cfg_100)
    jac_100 = evaluate_exact_jacobian_metrics(res_100.warp_fwd)

    # CFL 0.25 must strictly maintain folding < 0.1%
    assert jac_025['folding_percentage'] < 0.1, (
        f"CFL 0.25 folding rate {jac_025['folding_percentage']:.4f}% >= 0.1%"
    )
    # CFL 0.25 must yield greater or equal minimum determinant than CFL 1.0
    assert jac_025['min_jacobian'] >= jac_100['min_jacobian'] - 1e-5, (
        f"CFL 0.25 min det(J) ({jac_025['min_jacobian']:.4f}) degraded compared to CFL 1.0 ({jac_100['min_jacobian']:.4f})"
    )


if __name__ == '__main__':
    run_empirical_challenge_suite()
