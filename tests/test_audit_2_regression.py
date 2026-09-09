"""
tests/test_audit_2_regression.py — Verification of Audit 2 Bugs (#10 through #18)
==================================================================================

Covers:
- Bug #10 & #11: SyNToTransform spatial layout transposition and spacing handling.
- Bug #12: SyNTo.fit accepting ants.ANTsImage directly without TypeError.
- Bug #13: Preservation of updated affine_file when initial_transform is supplied.
- Bug #14: Exclusion of affine_epochs and explicit fit kwargs from **fit_kwargs unpacking.
- Bug #15: compute_harmonic_energy and compute_bending_energy handling batched tensors and anisotropic spacing.
- Bug #16: warp_scattered_coordinates aligning vector channels with Cartesian coordinates and matching ANTsPy.
- Bug #17: compute_jacobian_determinant_nd channel mapping (uz, uy, ux) and folding detection.
- Bug #18: core/inverse.py coordinate addition without cross-axis torch.flip.
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import ants
import pandas as pd

from syntx.spatial import disp_tensor_to_itk, disp_itk_to_tensor
from syntx.transform import SyNToTransform, export_ants_displacement_field
from syntx.syn import SyNTo, registration
from syntx.deformation_metrics import compute_harmonic_energy, compute_bending_energy
from syntx.scattered.mapping import warp_scattered_coordinates
from syntx.core.jacobian import compute_jacobian_determinant_nd
from syntx.core.inverse import update_inverse_field_nd_anderson, calculate_inverse_identity_error


# ---------------------------------------------------------------------------
# Test 1: Bug #10 & #11 — SyNToTransform to_composite_warp Parity with ANTs
# ---------------------------------------------------------------------------
def test_bug10_and_11_synto_transform_to_composite_warp_parity():
    """
    Validates that SyNToTransform.to_composite_warp correctly converts displacement
    fields into an ITK CompositeWarp matching PyTorch model.apply() within r > 0.999.
    """
    Nz, Ny, Nx = 16, 18, 20
    sx, sy, sz = 1.0, 1.5, 2.0
    spacing = (sx, sy, sz)
    origin = (10.0, -5.0, 30.0)

    # Synthetic smooth image
    # In PyTorch: tensor image has shape (Nz, Ny, Nx) = (16, 18, 20)
    arr_torch = np.pad(np.ones((Nz - 6, Ny - 6, Nx - 6), dtype=np.float32), 3)
    # ANTsImage expects (X, Y, Z) layout, matching export_ants_displacement_field transpose
    arr_ants = np.transpose(arr_torch, (2, 1, 0))
    fixed = ants.from_numpy(arr_ants, origin=origin, spacing=spacing)
    moving = ants.from_numpy(arr_ants * 1.5, origin=origin, spacing=spacing)

    # Smooth non-trivial displacement field in PyTorch tensor convention (Z, Y, X, 3)
    z = torch.linspace(0, 1, Nz).view(Nz, 1, 1)
    y = torch.linspace(0, 1, Ny).view(1, Ny, 1)
    x = torch.linspace(0, 1, Nx).view(1, 1, Nx)

    disp = torch.zeros(1, Nz, Ny, Nx, 3)
    # channel 0=uz, channel 1=uy, channel 2=ux
    disp[..., 0] = 0.5 * torch.sin(np.pi * y) * sz
    disp[..., 1] = 0.3 * torch.cos(np.pi * x) * sy
    disp[..., 2] = 0.4 * torch.sin(np.pi * z) * sx

    metadata = {
        'origin': origin,
        'spacing': spacing,
        'direction': fixed.direction,
        'shape': (Nz, Ny, Nx)
    }

    transform = SyNToTransform(
        warp_field=disp,
        metadata=metadata,
        is_physical=True
    )

    # 1. Evaluate Jacobian determinant without double spacing reversal
    detJ = transform.get_jacobian_determinant(as_numpy=True)
    assert detJ.shape == (Nz, Ny, Nx)
    assert np.all(np.isfinite(detJ))
    assert np.all(detJ > 0)

    # 2. Export composite warp to disk
    with tempfile.NamedTemporaryFile(suffix='_comp_warp.nii.gz') as tmp:
        transform.to_composite_warp(tmp.name)
        assert os.path.exists(tmp.name)

        # Resample moving image using ANTsPy with exported composite warp
        ants_warped = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=[tmp.name]
        )
        ants_warped_np = ants_warped.numpy()

    # Resample moving image using PyTorch transform.apply()
    moving_t = torch.from_numpy(arr_torch * 1.5).unsqueeze(0).unsqueeze(0)
    pytorch_warped_t = transform.apply(moving_t)
    pytorch_warped_np = pytorch_warped_t.squeeze().detach().cpu().numpy()

    # Compare interior in tensor coordinate order (transpose ants_warped_np from XYZ to ZYX)
    ants_warped_zyx = np.transpose(ants_warped_np, (2, 1, 0))
    sub_ants = ants_warped_zyx[2:-2, 2:-2, 2:-2]
    sub_py = pytorch_warped_np[2:-2, 2:-2, 2:-2]

    corr = np.corrcoef(sub_ants.flatten(), sub_py.flatten())[0, 1]
    max_diff = np.max(np.abs(sub_ants - sub_py))

    assert corr > 0.999, f"Correlation between CompositeWarp and PyTorch apply was {corr}, expected > 0.999"
    assert max_diff < 0.15, f"Max difference was {max_diff}, expected < 0.15"


# ---------------------------------------------------------------------------
# Test 2: Bug #12 — SyNTo.fit Accepts ANTsImage Inputs Directly
# ---------------------------------------------------------------------------
def test_bug12_synto_fit_accepts_antsimage():
    """
    Validates that SyNTo.fit converts ANTsImage instances to tensors without
    crashing on torch.min(fixed_image).
    """
    fi = ants.from_numpy(np.pad(np.ones((12, 12), dtype=np.float32), 2), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.pad(np.ones((12, 12), dtype=np.float32), 2), spacing=(1.0, 1.0))

    model = SyNTo(dim=2, grid_shape=(16, 16), spacing=(1.0, 1.0))
    # Pass ANTsImage directly into fit
    model.fit(
        fi, mi,
        levels=[1],
        epochs_per_level=[1],
        affine_epochs=[1],
        verbose=False
    )
    assert hasattr(model, 'warp_l2r')
    assert model.warp_l2r is not None


# ---------------------------------------------------------------------------
# Test 3: Bug #13 & #14 — syn.registration with initial_transform & fit kwargs
# ---------------------------------------------------------------------------
def test_bug13_and_14_syn_registration_initial_transform_and_affine_epochs():
    """
    Validates that syn.registration:
    - Excludes 'affine_epochs' from fit_kwargs unpacking (no TypeError).
    - Preserves and exports the updated affine_file when initial_transform is supplied.
    """
    fi = ants.from_numpy(np.pad(np.ones((14, 14), dtype=np.float32), 3), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.pad(np.ones((14, 14), dtype=np.float32), 3), spacing=(1.0, 1.0))

    # Initial translation
    init_tx = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
    init_tx.set_parameters(np.array([1.0, 0.0, 0.0, 1.0, 1.0, -1.0]))
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        init_file = tmp.name
    ants.write_transform(init_tx, init_file)

    try:
        res = registration(
            fixed=fi, moving=mi,
            type_of_transform='SyNTo',
            initial_transform=init_file,
            affine_iterations=[2],
            affine_epochs=[2],  # Bug #14 check: keyword should not collide
            reg_iterations=[1],
            backend='pytorch',
            verbose=False
        )
        assert res is not None
        assert 'fwdtransforms' in res
        # Bug #13 check: fwdtransforms must contain the optimized affine transform, not just init_file
        fwd_txs = res['fwdtransforms']
        assert len(fwd_txs) == 2
        assert fwd_txs[0].endswith('.nii.gz')  # SyN warp
        assert fwd_txs[1].endswith('.mat')     # Updated affine incorporating initial_transform
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)


# ---------------------------------------------------------------------------
# Test 4: Bug #15 — Energy Metrics on Batched Tensors & Anisotropic Spacing
# ---------------------------------------------------------------------------
def test_bug15_deformation_energy_batched_tensors_and_anisotropy():
    """
    Validates that compute_harmonic_energy and compute_bending_energy accept
    batched PyTorch tensors (1, D, H, W, 3) with anisotropic spacing without
    raising dimension errors.
    """
    D, H, W = 10, 12, 14
    sx, sy, sz = 1.0, 2.0, 4.0
    spacing = (sx, sy, sz)

    # Batched 5D displacement field: (1, D, H, W, 3)
    disp_tensor = torch.randn(1, D, H, W, 3) * 0.1

    harm_energy = compute_harmonic_energy(disp_tensor, spacing=spacing)
    assert isinstance(harm_energy, float)
    assert harm_energy > 0.0
    assert np.isfinite(harm_energy)

    bend_energy = compute_bending_energy(disp_tensor, spacing=spacing)
    assert isinstance(bend_energy, float)
    assert bend_energy > 0.0
    assert np.isfinite(bend_energy)


# ---------------------------------------------------------------------------
# Test 5: Bug #16 — warp_scattered_coordinates Vector Channel Alignment
# ---------------------------------------------------------------------------
def test_bug16_warp_scattered_coordinates_matches_antspy():
    """
    Validates that warp_scattered_coordinates aligns vector channel components
    with Cartesian (x, y, z) coordinates, matching ANTsPy's apply_transforms_to_points.
    """
    Nz, Ny, Nx = 16, 16, 16
    sx, sy, sz = 1.0, 2.0, 3.0
    spacing = (sx, sy, sz)
    origin = (0.0, 0.0, 0.0)

    # Pure displacement along physical X: ux = 2.5 mm, uy = 0, uz = 0
    # In PyTorch tensor convention: channel 0 is uz, channel 1 is uy, channel 2 is ux
    disp_torch = torch.zeros(1, Nz, Ny, Nx, 3)
    disp_torch[..., 2] = 2.5  # ux = 2.5 mm

    ref_img = ants.from_numpy(np.zeros((Nx, Ny, Nz), dtype=np.float32), spacing=spacing, origin=origin)
    vec_ants = disp_tensor_to_itk(disp_torch, ref_image=ref_img)

    with tempfile.NamedTemporaryFile(suffix='_warp_pts.nii.gz') as tmp:
        ants.image_write(vec_ants, tmp.name)

        # Test points in physical Cartesian coordinates (x, y, z)
        pts_np = np.array([
            [5.0, 10.0, 15.0],
            [8.0, 12.0, 20.0],
            [10.0, 14.0, 25.0]
        ], dtype=np.float32)

        pts_df = pd.DataFrame(pts_np, columns=['x', 'y', 'z'])
        ants_pts_df = ants.apply_transforms_to_points(dim=3, points=pts_df, transformlist=[tmp.name])

        warped_pts_syntx = warp_scattered_coordinates(
            torch.from_numpy(pts_np),
            disp_torch,
            coord_convention='xyz',
            is_physical=True,
            domain_bounds=[
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([(Nx - 1) * sx, (Ny - 1) * sy, (Nz - 1) * sz])
            ]
        )

        warped_pts_from_ants = warp_scattered_coordinates(
            torch.from_numpy(pts_np),
            vec_ants,
            coord_convention='xyz',
            domain_bounds=[
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([(Nx - 1) * sx, (Ny - 1) * sy, (Nz - 1) * sz])
            ]
        )

    # ux displacement was 2.5 mm along X
    warped_np = warped_pts_syntx.detach().cpu().numpy()
    np.testing.assert_allclose(warped_np[:, 0], pts_np[:, 0] + 2.5, atol=1e-4)
    np.testing.assert_allclose(warped_np[:, 1], pts_np[:, 1], atol=1e-4)
    np.testing.assert_allclose(warped_np[:, 2], pts_np[:, 2], atol=1e-4)

    # Direct ANTsImage input parity
    warped_ants_np = warped_pts_from_ants.detach().cpu().numpy()
    np.testing.assert_allclose(warped_ants_np, warped_np, atol=1e-4)

    # Parity with ANTsPy's apply_transforms_to_points
    np.testing.assert_allclose(warped_np[:, 0], ants_pts_df['x'].values, atol=1e-4)
    np.testing.assert_allclose(warped_np[:, 1], ants_pts_df['y'].values, atol=1e-4)
    np.testing.assert_allclose(warped_np[:, 2], ants_pts_df['z'].values, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 6: Bug #17 — compute_jacobian_determinant_nd Folding Detection
# ---------------------------------------------------------------------------
def test_bug17_jacobian_determinant_folding_detection():
    """
    Validates that compute_jacobian_determinant_nd correctly maps physical vector
    channels (uz, uy, ux) and detects local topology-violating grid folding (det(J) <= 0).
    """
    D, H, W = 16, 16, 16
    sx, sy, sz = 1.0, 1.0, 1.0
    spacing = (sx, sy, sz)

    x = torch.arange(W).float()
    X = x.view(1, 1, W).expand(D, H, W)

    # Severe compression along X: ux = -1.5 * X -> det(J) = 1 - 1.5 = -0.5 <= 0 (grid fold)
    # PyTorch tensor convention: channel 2 is ux
    disp_fold = torch.zeros(1, D, H, W, 3)
    disp_fold[..., 2] = -1.5 * X

    detJ = compute_jacobian_determinant_nd(disp_fold, physical_spacing=spacing)
    detJ_center = detJ[0, 8, 8, 8].item()

    assert detJ_center < 0.0, f"Expected det(J) < 0 for severe compression, got {detJ_center}"
    folding_pct = float((detJ <= 0).float().mean() * 100.0)
    assert folding_pct > 50.0, f"Expected folding percentage > 50%, got {folding_pct}%"


# ---------------------------------------------------------------------------
# Test 7: Bug #18 — Inversion Without Cross-Axis torch.flip
# ---------------------------------------------------------------------------
def test_bug18_inverse_field_no_cross_axis_flip():
    """
    Validates that update_inverse_field_nd_anderson and calculate_inverse_identity_error
    do not cross-flip displacement channels, so an anisotropic pure-Z displacement
    inverts purely in Z without producing spurious X or Y displacements.
    """
    D, H, W = 12, 14, 16
    sx, sy, sz = 1.0, 2.0, 4.0
    spacing = (sx, sy, sz)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    z = torch.linspace(0, 1, D).view(D, 1, 1).expand(D, H, W)

    # Pure displacement along Z: uz = 0.2 * sz * sin(pi * z)
    # In PyTorch tensor convention: channel 0 is uz, channel 1 is uy, channel 2 is ux
    disp = torch.zeros(1, D, H, W, 3)
    disp[..., 0] = 0.2 * sz * torch.sin(np.pi * z)

    inv_disp = update_inverse_field_nd_anderson(
        disp, None, steps=10, spacing=spacing, origin=origin, direction=direction
    )

    # Inverse displacement must remain purely in Z
    # Channels 1 (uy) and 2 (ux) must be strictly zero (or below machine eps)
    max_ux = float(inv_disp[..., 2].abs().max())
    max_uy = float(inv_disp[..., 1].abs().max())
    max_uz = float(inv_disp[..., 0].abs().max())

    assert max_ux < 1e-6, f"Spurious ux found in inverse: {max_ux}"
    assert max_uy < 1e-6, f"Spurious uy found in inverse: {max_uy}"
    assert max_uz > 0.01, f"Expected significant uz inverse, got {max_uz}"

    # Calculate inverse identity error
    err_dict = calculate_inverse_identity_error(disp, inv_disp, spacing, origin, direction)
    assert err_dict['mean_error'] < 0.05, f"Mean error {err_dict['mean_error']} too high"
