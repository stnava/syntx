"""
Adversarial Verification Suite for Milestone 3:
TVF Layout Changes, Velocity Parameter Updates, and Downstream Solver Invariance.

Verification targets:
1. TVFModel.velocity in-place updates via `with torch.no_grad(): self.velocity.sub_(...)`
   confirming `_version` increments and invalidates `_v_max_cache` across optimizer settings (CFL, Adam, momentum).
2. TVF gradient resampling contiguity invariance on anisotropic 2D and 3D volumes with oblique direction matrices
   (verifying L_inf < 1e-6 numerical parity against legacy .contiguous() calls).
3. SyN Eulerian composition and Lagrangian pullback contiguity invariance on anisotropic 3D volumes.
4. End-to-end TVF deformation regularity (min det(J) > 0, folding = 0.0%) and inverse consistency.
5. End-to-end SyN deformation regularity (min det(J) > 0, folding = 0.0%) and inverse consistency.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import ants

from syntx.tvf import TVFModel
from syntx.syn import SyNTo
from syntx import tvf, syn
from syntx.spatial import (
    jacobian_determinant,
    physical_to_normalized_torch_cached,
)
from syntx.core.grid import grid_sample_nd


# ==============================================================================
# 1. TVF Velocity Version Tracking & Cache Invalidation Stress Tests
# ==============================================================================

def test_tvf_velocity_sub_inplace_increments_version_and_invalidates_cache():
    """
    Test that updating velocity via `with torch.no_grad(): self.velocity.sub_(...)`
    increments self.velocity._version and invalidates _v_max_cache entries.
    """
    model = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=3)
    target_shape = (32, 32)
    
    # Populate initial cache
    v_max_0 = model._get_v_max_voxel(model.velocity, target_shape)
    v0_version = model.velocity._version
    key_0 = (id(model.velocity), v0_version, target_shape)
    assert key_0 in model._v_max_cache
    assert v_max_0 == 0.0

    # Perform in-place update using modernized pattern
    update = torch.ones_like(model.velocity) * 1.5
    with torch.no_grad():
        model.velocity.sub_(update)

    v1_version = model.velocity._version
    assert v1_version == v0_version + 1, f"Version expected {v0_version + 1}, got {v1_version}"

    # Key for new version must not yet be in cache
    key_1 = (id(model.velocity), v1_version, target_shape)
    assert key_1 not in model._v_max_cache

    # Querying _get_v_max_voxel must compute new value and populate cache under key_1
    v_max_1 = model._get_v_max_voxel(model.velocity, target_shape)
    assert key_1 in model._v_max_cache
    assert v_max_1 > 0.0, f"Expected positive v_max, got {v_max_1}"
    assert math.isclose(v_max_1, 1.5 * math.sqrt(2), rel_tol=1e-4)

    # Subsequent identical call reuses cache
    assert model._get_v_max_voxel(model.velocity, target_shape) == v_max_1


@pytest.mark.parametrize("cfl_momentum", [0.0, 0.5, 0.9])
def test_tvf_fit_cfl_version_tracking_across_epochs(cfl_momentum):
    """
    Test that TVFModel.fit() with optimizer_type='cfl' and elastic_sigma=0.0
    increments velocity._version on EVERY epoch and invalidates _v_max_cache.
    """
    fixed_np = np.zeros((32, 32), dtype=np.float32)
    moving_np = np.zeros((32, 32), dtype=np.float32)
    fixed_np[10:22, 10:22] = 1.0
    moving_np[12:24, 10:22] = 1.0  # shifted by 2 voxels

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    fi_t = torch.tensor(fixed_np, device=device).unsqueeze(0).unsqueeze(0)
    mi_t = torch.tensor(moving_np, device=device).unsqueeze(0).unsqueeze(0)

    model = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=3, solver='rk4').to(device)
    
    n_epochs = 3
    initial_version = model.velocity._version

    model.fit(
        fi_t,
        mi_t,
        levels=[1],
        epochs_per_level=[n_epochs],
        affine_epochs=0,
        optimizer_type='cfl',
        elastic_sigma=0.0,
        cfl_momentum=cfl_momentum,
        verbose=False,
    )

    final_version = model.velocity._version
    assert final_version >= initial_version + n_epochs, (
        f"Expected version >= {initial_version + n_epochs}, but got {final_version}"
    )

    current_key = (id(model.velocity), final_version, (32, 32))
    assert current_key in model._v_max_cache, "Current velocity version not found in _v_max_cache!"
    v_max_current = model._v_max_cache[current_key]
    assert v_max_current > 0.0, "v_max_voxel should be non-zero after optimization"


def test_tvf_fit_adam_version_tracking():
    """
    Test that TVFModel.fit() with optimizer_type='adam' and elastic_sigma > 0
    also properly increments velocity._version and caches v_max.
    """
    fixed_np = np.zeros((24, 24), dtype=np.float32)
    moving_np = np.zeros((24, 24), dtype=np.float32)
    fixed_np[8:16, 8:16] = 1.0
    moving_np[10:18, 8:16] = 1.0

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    fi_t = torch.tensor(fixed_np, device=device).unsqueeze(0).unsqueeze(0)
    mi_t = torch.tensor(moving_np, device=device).unsqueeze(0).unsqueeze(0)

    model = TVFModel(dim=2, image_shape=(24, 24), velocity_shape=(12, 12), n_time_steps=3).to(device)
    v0 = model.velocity._version

    model.fit(
        fi_t,
        mi_t,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=0,
        optimizer_type='adam',
        elastic_sigma=1.0,
        verbose=False,
    )

    v1 = model.velocity._version
    assert v1 >= v0 + 2, f"Expected version increment of at least 2, got {v1 - v0}"
    assert (id(model.velocity), v1, (24, 24)) in model._v_max_cache


# ==============================================================================
# 2. TVF Gradient Resampling Contiguity Invariance Tests (Anisotropic & Oblique)
# ==============================================================================

def test_tvf_gradient_resampling_anisotropic_2d():
    """
    Stress-test TVF spatial gradient resampling contiguity invariance on a 2D anisotropic
    grid with non-identity direction cosine matrix.
    """
    H, W = 23, 37
    
    # 2D rotation matrix (30 degrees)
    theta = np.deg2rad(30.0)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=np.float32)
    dir_mat = torch.tensor(rot_mat)

    torch.manual_seed(42)
    grad_I_curr = torch.randn(1, H, W, 2)
    phi_fixed_norm = torch.rand(1, H, W, 2) * 1.8 - 0.9
    g_im = torch.randn(1, 1, H, W)

    grad_J_curr = torch.randn(1, H, W, 2)
    phi_moving_norm = torch.rand(1, H, W, 2) * 1.8 - 0.9
    g_jm = torch.randn(1, 1, H, W)
    M_phys_zyx = torch.randn(2, 2)

    # 1. With .contiguous() (baseline legacy pattern)
    grad_I_mid_c = grid_sample_nd(
        grad_I_curr.movedim(-1, 1).contiguous(), phi_fixed_norm.contiguous(),
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1).contiguous()
    grad_I_mid_c = torch.matmul(grad_I_mid_c, dir_mat)

    grad_J_mid_c = grid_sample_nd(
        grad_J_curr.movedim(-1, 1).contiguous(), phi_moving_norm.contiguous(),
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1).contiguous()
    grad_J_mid_c = torch.matmul(grad_J_mid_c, dir_mat)
    grad_J_mid_c = torch.matmul(grad_J_mid_c, M_phys_zyx)

    grad_wrt_phi_fixed_c = (g_im.movedim(1, -1).contiguous() * grad_I_mid_c).contiguous()
    grad_wrt_phi_moving_c = (g_jm.movedim(1, -1).contiguous() * grad_J_mid_c).contiguous()
    combined_grad_c = (grad_wrt_phi_moving_c - grad_wrt_phi_fixed_c) / 2.0

    # 2. Without .contiguous() (remediated worker_m3 pattern)
    grad_I_mid_nc = grid_sample_nd(
        grad_I_curr.movedim(-1, 1), phi_fixed_norm,
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1)
    grad_I_mid_nc = torch.matmul(grad_I_mid_nc, dir_mat)

    grad_J_mid_nc = grid_sample_nd(
        grad_J_curr.movedim(-1, 1), phi_moving_norm,
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1)
    grad_J_mid_nc = torch.matmul(grad_J_mid_nc, dir_mat)
    grad_J_mid_nc = torch.matmul(grad_J_mid_nc, M_phys_zyx)

    grad_wrt_phi_fixed_nc = g_im.movedim(1, -1) * grad_I_mid_nc
    grad_wrt_phi_moving_nc = g_jm.movedim(1, -1) * grad_J_mid_nc
    combined_grad_nc = (grad_wrt_phi_moving_nc - grad_wrt_phi_fixed_nc) / 2.0

    # Numerical assertions: sampling is bitwise identical, matmuls are within 1e-6
    assert torch.equal(grad_I_mid_c, grad_I_mid_nc), "grad_I_mid differs without .contiguous()"
    max_diff_J = (grad_J_mid_c - grad_J_mid_nc).abs().max().item()
    assert max_diff_J < 1e-6, f"grad_J_mid max diff too large: {max_diff_J}"
    max_diff_comb = (combined_grad_c - combined_grad_nc).abs().max().item()
    assert max_diff_comb < 1e-6, f"combined_grad max diff too large: {max_diff_comb}"


def test_tvf_gradient_resampling_anisotropic_3d_oblique():
    """
    Stress-test TVF spatial gradient resampling contiguity invariance on a 3D anisotropic
    volume with arbitrary 3D rotation direction matrix.
    """
    D, H, W = 15, 23, 31

    # Construct 3D orthogonal rotation matrix (yaw-pitch-roll)
    alpha, beta, gamma = np.deg2rad(15.0), np.deg2rad(25.0), np.deg2rad(35.0)
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ], dtype=np.float32)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ], dtype=np.float32)
    rot_3d = Rz @ Ry @ Rx
    dir_mat = torch.tensor(rot_3d)

    torch.manual_seed(123)
    grad_I_curr = torch.randn(1, D, H, W, 3)
    phi_fixed_norm = torch.rand(1, D, H, W, 3) * 1.8 - 0.9
    g_im = torch.randn(1, 1, D, H, W)

    grad_J_curr = torch.randn(1, D, H, W, 3)
    phi_moving_norm = torch.rand(1, D, H, W, 3) * 1.8 - 0.9
    g_jm = torch.randn(1, 1, D, H, W)
    M_phys_zyx = torch.randn(3, 3)

    # 1. Contiguous
    grad_I_mid_c = grid_sample_nd(
        grad_I_curr.movedim(-1, 1).contiguous(), phi_fixed_norm.contiguous(),
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1).contiguous()
    grad_I_mid_c = torch.matmul(grad_I_mid_c, dir_mat)

    grad_J_mid_c = grid_sample_nd(
        grad_J_curr.movedim(-1, 1).contiguous(), phi_moving_norm.contiguous(),
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1).contiguous()
    grad_J_mid_c = torch.matmul(grad_J_mid_c, dir_mat)
    grad_J_mid_c = torch.matmul(grad_J_mid_c, M_phys_zyx)

    grad_wrt_phi_fixed_c = (g_im.movedim(1, -1).contiguous() * grad_I_mid_c).contiguous()
    grad_wrt_phi_moving_c = (g_jm.movedim(1, -1).contiguous() * grad_J_mid_c).contiguous()
    combined_grad_c = (grad_wrt_phi_moving_c - grad_wrt_phi_fixed_c) / 2.0

    # 2. Non-contiguous
    grad_I_mid_nc = grid_sample_nd(
        grad_I_curr.movedim(-1, 1), phi_fixed_norm,
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1)
    grad_I_mid_nc = torch.matmul(grad_I_mid_nc, dir_mat)

    grad_J_mid_nc = grid_sample_nd(
        grad_J_curr.movedim(-1, 1), phi_moving_norm,
        mode='bilinear', padding_mode='zeros'
    ).movedim(1, -1)
    grad_J_mid_nc = torch.matmul(grad_J_mid_nc, dir_mat)
    grad_J_mid_nc = torch.matmul(grad_J_mid_nc, M_phys_zyx)

    grad_wrt_phi_fixed_nc = g_im.movedim(1, -1) * grad_I_mid_nc
    grad_wrt_phi_moving_nc = g_jm.movedim(1, -1) * grad_J_mid_nc
    combined_grad_nc = (grad_wrt_phi_moving_nc - grad_wrt_phi_fixed_nc) / 2.0

    # Numerical assertions
    assert (grad_I_mid_c - grad_I_mid_nc).abs().max().item() < 1e-6
    assert (grad_J_mid_c - grad_J_mid_nc).abs().max().item() < 1e-6
    assert (grad_wrt_phi_fixed_c - grad_wrt_phi_fixed_nc).abs().max().item() < 1e-6
    assert (grad_wrt_phi_moving_c - grad_wrt_phi_moving_nc).abs().max().item() < 1e-6
    assert (combined_grad_c - combined_grad_nc).abs().max().item() < 1e-6


def test_syn_composition_anisotropic_3d_contiguity():
    """
    Stress-test SyN Eulerian composition and Lagrangian pullback contiguity invariance
    on anisotropic 3D volumes.
    """
    shape = (16, 24, 32)
    spacing = (0.75, 1.25, 2.0)
    origin = (0.0, 0.0, 0.0)
    direction = torch.eye(3)

    shape_t = torch.tensor(shape, dtype=torch.float32)
    spacing_t = torch.tensor(spacing, dtype=torch.float32)
    origin_t = torch.tensor(origin, dtype=torch.float32)
    direction_t = direction

    torch.manual_seed(99)
    warp = torch.randn(1, *shape, 3) * 0.5
    delta = torch.randn(1, *shape, 3) * 0.1

    from syntx.spatial import get_physical_grid_torch
    coords_phys = get_physical_grid_torch(shape, spacing, origin, direction) + warp
    coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)

    # 1. Eulerian composition with .contiguous()
    delta_pb_c = F.grid_sample(
        delta.movedim(-1, 1).contiguous(), coords_norm.contiguous(),
        padding_mode='border', align_corners=True
    ).movedim(1, -1).contiguous()

    # 2. Eulerian composition without .contiguous() (remediated)
    delta_pb_nc = F.grid_sample(
        delta.movedim(-1, 1), coords_norm,
        padding_mode='border', align_corners=True
    ).movedim(1, -1)

    assert torch.equal(delta_pb_c, delta_pb_nc), "SyN Eulerian pullback differs without .contiguous()"

    # 3. Lagrangian pullback with vs without .contiguous()
    warp_sampled_c = F.grid_sample(
        warp.movedim(-1, 1).contiguous(), coords_norm.contiguous(),
        padding_mode='border', align_corners=True
    ).movedim(1, -1).contiguous()

    warp_sampled_nc = F.grid_sample(
        warp.movedim(-1, 1), coords_norm,
        padding_mode='border', align_corners=True
    ).movedim(1, -1)

    assert torch.equal(warp_sampled_c, warp_sampled_nc), "SyN Lagrangian composition differs"


# ==============================================================================
# 3. Downstream Registration Quality: Regularity, Folding, & Inverse Consistency
# ==============================================================================

def _create_synthetic_sphere_pair_3d(shape=(24, 24, 24)):
    """Generate a pair of 3D synthetic sphere images with known deformation."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center1 = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    center2 = (shape[0] // 2 + 1, shape[1] // 2 - 1, shape[2] // 2)

    r1 = 6.0
    r2 = 7.0
    dist1 = np.sqrt((z - center1[0])**2 + (y - center1[1])**2 + (x - center1[2])**2)
    dist2 = np.sqrt((z - center2[0])**2 + (y - center2[1])**2 + (x - center2[2])**2)

    fixed_arr = np.clip(1.0 - (dist1 / r1)**2, 0.0, 1.0).astype(np.float32)
    moving_arr = np.clip(1.0 - (dist2 / r2)**2, 0.0, 1.0).astype(np.float32)

    fixed = ants.from_numpy(fixed_arr, spacing=(1.0, 1.0, 1.0))
    moving = ants.from_numpy(moving_arr, spacing=(1.0, 1.0, 1.0))
    return fixed, moving


def test_tvf_e2e_regularity_and_inverse_consistency_2d():
    """
    End-to-end 2D TVF registration test verifying:
    - min det(J) > 0 (strictly positive Jacobian determinant everywhere)
    - grid folding == 0.0%
    - physical inverse consistency error is small and bounded
    - overlap DICE improves
    """
    f_arr = np.zeros((32, 32), dtype=np.float32)
    m_arr = np.zeros((32, 32), dtype=np.float32)
    f_arr[8:24, 8:24] = 1.0
    m_arr[10:26, 7:23] = 1.0

    fixed = ants.from_numpy(f_arr, spacing=(1.0, 1.0))
    moving = ants.from_numpy(m_arr, spacing=(1.0, 1.0))

    initial_dice = float((2.0 * (f_arr * m_arr).sum()) / (f_arr.sum() + m_arr.sum() + 1e-8))

    reg = tvf(
        fixed=fixed,
        moving=moving,
        levels=[2, 1],
        reg_iterations=[5, 5],
        affine_iterations=0,
        n_time_steps=3,
        solver='rk4',
        optimizer='cfl',
        elastic_sigma=0.0,
        flow_sigma=1.0,
        verbose=False,
    )

    fwd_warp_path = [t for t in reg['fwdtransforms'] if t.endswith('.nii.gz')][0]

    # Check Jacobian determinant regularity
    jac_img = ants.create_jacobian_determinant_image(fixed, fwd_warp_path, do_log=False)
    jac_np = jac_img.numpy()

    min_det = float(jac_np.min())
    folding_pct = float((jac_np <= 0).sum() / jac_np.size * 100.0)

    assert min_det > 0.0, f"TVF min det(J) must be > 0, got {min_det}"
    assert folding_pct == 0.0, f"TVF folding must be 0.0%, got {folding_pct}%"

    # Check physical inverse consistency error
    assert 'inverse_identity_errors' in reg, "reg missing inverse_identity_errors"
    inv_err_info = reg['inverse_identity_errors']['phi_1']
    mean_err = float(inv_err_info['mean_error'])
    max_err = float(inv_err_info['max_error'])

    assert mean_err < 0.25, f"Mean inverse consistency error too high: {mean_err:.4f} mm"
    assert max_err < 1.0, f"Max inverse consistency error too high: {max_err:.4f} mm"

    # Warped image check
    warped_arr = reg['warpedmovout'].numpy()
    warped_bin = (warped_arr > 0.5).astype(np.float32)
    final_dice = float((2.0 * (f_arr * warped_bin).sum()) / (f_arr.sum() + warped_bin.sum() + 1e-8))
    assert final_dice >= initial_dice, f"Final DICE {final_dice} should be >= initial DICE {initial_dice}"


def test_tvf_e2e_regularity_and_inverse_consistency_3d():
    """
    End-to-end 3D TVF registration test verifying:
    - min det(J) > 0
    - folding == 0.0%
    - bounded inverse consistency error
    """
    fixed, moving = _create_synthetic_sphere_pair_3d(shape=(24, 24, 24))

    reg = tvf(
        fixed=fixed,
        moving=moving,
        levels=[2, 1],
        reg_iterations=[3, 3],
        affine_iterations=0,
        n_time_steps=3,
        solver='rk4',
        optimizer='cfl',
        elastic_sigma=0.0,
        flow_sigma=1.0,
        verbose=False,
    )

    fwd_warp_path = [t for t in reg['fwdtransforms'] if t.endswith('.nii.gz')][0]

    jac_img = ants.create_jacobian_determinant_image(fixed, fwd_warp_path, do_log=False)
    jac_np = jac_img.numpy()

    min_det = float(jac_np.min())
    folding_pct = float((jac_np <= 0).sum() / jac_np.size * 100.0)

    assert min_det > 0.0, f"3D TVF min det(J) must be > 0, got {min_det}"
    assert folding_pct == 0.0, f"3D TVF folding must be 0.0%, got {folding_pct}%"

    inv_err_info = reg['inverse_identity_errors']['phi_1']
    mean_err = float(inv_err_info['mean_error'])
    assert mean_err < 0.25, f"3D TVF mean inverse error too high: {mean_err:.4f} mm"


def test_syn_e2e_regularity_and_inverse_consistency_3d():
    """
    End-to-end 3D SyN registration test verifying:
    - min det(J) > 0
    - folding == 0.0%
    - in-place Adam updates produce regular diffeomorphic warps
    """
    fixed, moving = _create_synthetic_sphere_pair_3d(shape=(24, 24, 24))

    reg = syn(
        fixed=fixed,
        moving=moving,
        levels=[2, 1],
        epochs=[3, 3],
        affine_iterations=0,
        grad_step=0.2,
        flow_sigma=1.5,
        total_sigma=0.0,
        formulation='eulerian',
        use_analytical_gradients=False,
        verbose=False,
    )

    fwd_warp_path = [t for t in reg['fwdtransforms'] if t.endswith('.nii.gz')][0]

    jac_img = ants.create_jacobian_determinant_image(fixed, fwd_warp_path, do_log=False)
    jac_np = jac_img.numpy()

    min_det = float(jac_np.min())
    folding_pct = float((jac_np <= 0).sum() / jac_np.size * 100.0)

    assert min_det > 0.0, f"3D SyN min det(J) must be > 0, got {min_det}"
    assert folding_pct == 0.0, f"3D SyN folding must be 0.0%, got {folding_pct}%"

    inv_err_info = reg['inverse_identity_errors']['phi_1']
    mean_err = float(inv_err_info['mean_error'])
    assert mean_err < 0.25, f"3D SyN mean inverse error too high: {mean_err:.4f} mm"
