"""
Adversarial test suite challenging SyNToTransform methods (Milestone 2).
"""

import tempfile
import os
import numpy as np
import pytest
import torch
import ants
from scipy.spatial.transform import Rotation

import syntx.spatial as sp
from syntx.transform import SyNToTransform, create_ants_affine, export_ants_affine_transform
from syntx.core.inverse import update_inverse_field_nd


class TestAdversarialTransformM2:
    """Adversarial stress harness for SyNToTransform methods in M2."""

    def test_invert_with_ants_transform_2d_and_3d(self):
        """Task 1 Challenge: invert() must support ants.ANTsTransform in 2D and 3D without crashing."""
        for dim in [2, 3]:
            shape = (16, 20) if dim == 2 else (12, 14, 16)
            spacing = (0.8, 1.2) if dim == 2 else (0.8, 1.2, 1.5)
            origin = (1.0, 2.0) if dim == 2 else (1.0, 2.0, 3.0)
            direction = np.eye(dim, dtype=np.float32)
            meta = {'shape': shape, 'spacing': spacing, 'origin': origin, 'direction': direction}

            M = np.eye(dim, dtype=np.float32)
            t = np.array([2.0, -1.0, 3.0][:dim], dtype=np.float32)
            ants_tx = create_ants_affine(M, t, dim=dim)

            warp = torch.zeros(1, *shape, dim)
            tx = SyNToTransform(warp_field=warp, metadata=meta, affine_matrix=ants_tx)

            # This call crashes if ants.invert_transform is used instead of ants.invert_ants_transform
            inv_tx = tx.invert()
            assert inv_tx is not None
            assert isinstance(inv_tx.affine_matrix, ants.ANTsTransform)

            # Test mathematical roundtrip
            pt = (5.0, 6.0) if dim == 2 else (5.0, 6.0, 7.0)
            pt_fwd = tx.affine_matrix.apply_to_point(pt)
            pt_back = inv_tx.affine_matrix.apply_to_point(pt_fwd)
            assert np.allclose(pt, pt_back, atol=1e-4)

    def test_export_t_grid_axis_orientation_parity(self):
        """Task 1 & 2 Challenge: Exporting T_grid must NOT swap physical axes (X <-> Z in 3D, X <-> Y in 2D)."""
        # 3D test
        shape = (10, 20, 30)  # Nz=10, Ny=20, Nx=30
        spacing = (1.0, 2.0, 3.0)  # sx=1.0, sy=2.0, sz=3.0
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)
        meta = {'shape': shape, 'spacing': spacing, 'origin': origin, 'direction': direction}
        warp = torch.zeros(1, *shape, 3)

        # Shift along normalized grid X
        T_grid = torch.eye(4)[:3, :].unsqueeze(0)
        T_grid[0, 0, 3] = 0.5

        fixed_img = ants.from_numpy(np.zeros((30, 20, 10), dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        _, t_expected = sp.grid_to_physical_affine(T_grid[0].numpy(), fixed_img, fixed_img)

        tx = SyNToTransform(warp_field=warp, metadata=meta, T_grid=T_grid)
        with tempfile.TemporaryDirectory() as tmpdir:
            res = tx.export(outprefix=f"{tmpdir}/t_")
            tx_read = ants.read_transform(res['affine'])
            t_exported = tx_read.parameters[9:]

            # In ITK XYZ order, t_exported must match t_expected
            assert np.allclose(t_exported, t_expected, atol=1e-3), (
                f"3D Axis swap detected! Exported: {t_exported}, Expected: {t_expected}"
            )

    def test_invert_preserves_affine_when_initialized_via_t_grid(self):
        """Task 1 Challenge: invert() must preserve the affine transform when initialized via T_grid."""
        shape = (16, 16)
        warp = torch.zeros(1, *shape, 2)
        meta = {'origin': (0.0, 0.0), 'spacing': (1.0, 1.0), 'direction': np.eye(2), 'shape': shape}
        T_grid = torch.eye(3)[:2, :].unsqueeze(0)
        T_grid[0, 0, 2] = 0.4

        tx = SyNToTransform(warp_field=warp, metadata=meta, T_grid=T_grid)
        inv_tx = tx.invert()

        # Inverted transform must retain an affine component (either T_grid or affine_matrix)
        has_affine = (inv_tx.T_grid is not None) or (inv_tx.affine_matrix is not None) or (inv_tx.affine_grid is not None)
        assert has_affine, "invert() dropped the affine transform component from T_grid!"

    def test_apply_and_to_displacement_field_respect_affine_matrix(self):
        """Task 1 & 2 Challenge: apply() and to_displacement_field() must incorporate affine_matrix."""
        shape = (16, 16)
        warp = torch.zeros(1, *shape, 2)
        meta = {'origin': (0.0, 0.0), 'spacing': (1.0, 1.0), 'direction': np.eye(2), 'shape': shape}
        M = np.eye(2, dtype=np.float32)
        t = np.array([4.0, 0.0], dtype=np.float32)  # 4mm shift in X

        tx = SyNToTransform(warp_field=warp, metadata=meta, affine_matrix=(M, t))
        disp_img = tx.to_displacement_field()
        max_disp = np.max(np.abs(disp_img.numpy()))
        assert max_disp > 1.0, f"to_displacement_field() ignored affine_matrix! max_disp={max_disp}"

    def test_t_grid_displacement_field_default_mode(self):
        """Task 2 Challenge: to_displacement_field() must reflect T_grid under default is_physical=False."""
        shape = (16, 16)
        warp = torch.zeros(1, *shape, 2)
        meta = {'origin': (0.0, 0.0), 'spacing': (1.0, 1.0), 'direction': np.eye(2), 'shape': shape}
        T_grid = torch.eye(3)[:2, :].unsqueeze(0)
        T_grid[0, 0, 2] = 0.5  # shift in grid

        tx = SyNToTransform(warp_field=warp, metadata=meta, T_grid=T_grid, is_physical=False)
        disp = tx.to_displacement_field()
        max_disp = np.max(np.abs(disp.numpy()))
        assert max_disp > 1.0, f"to_displacement_field() ignored T_grid under is_physical=False! max_disp={max_disp}"

    def test_invert_physical_warp_passes_metadata_to_solver(self):
        """Task 1 Challenge: invert() must pass spatial metadata to update_inverse_field_nd when is_physical=True."""
        shape = (20, 20)
        spacing = (2.5, 0.5)
        meta = {'origin': (0.0, 0.0), 'spacing': spacing, 'direction': np.eye(2), 'shape': shape}
        warp_phys = torch.zeros(1, *shape, 2)
        warp_phys[0, 8:12, 8:12, 0] = 1.0
        warp_phys[0, 8:12, 8:12, 1] = 0.5

        tx = SyNToTransform(warp_field=warp_phys, metadata=meta, is_physical=True)
        inv_tx = tx.invert()

        # Expected inverse computed with proper physical metadata
        expected_inv = update_inverse_field_nd(warp_phys, spacing=spacing, origin=(0.0, 0.0), direction=np.eye(2))
        err = torch.max(torch.abs(inv_tx.warp_field - expected_inv)).item()
        assert err < 1e-3, f"invert() did not account for physical metadata! Discrepancy={err}"

    def test_normalized_to_physical_disp_exact_numerical_identity(self):
        """Task 3 Verification: normalized_to_physical_disp achieves exact numerical identity with legacy."""
        def legacy_calc(disp, target_shape, metadata, dim, device='cpu'):
            spatial_shape = torch.tensor(list(reversed(target_shape)), dtype=disp.dtype, device=device)
            voxel_disp = disp * (spatial_shape - 1.0) / 2.0
            direction = torch.tensor(metadata['direction'], dtype=disp.dtype, device=device)
            spacing = torch.tensor(metadata['spacing'], dtype=disp.dtype, device=device)
            phys_disp = voxel_disp * spacing
            phys_disp_flat = phys_disp.reshape(-1, dim)
            phys_disp_flat = phys_disp_flat @ direction.t()
            phys_disp = phys_disp_flat.reshape(disp.shape)
            return torch.flip(phys_disp, dims=[-1])

        # 1. Exact dyadic powers of 2 (exact 0.0 L_inf)
        shape_dyadic = (17, 33, 65)
        spacing_dyadic = (1.0, 2.0, 4.0)
        direction_dyadic = np.eye(3)
        disp_dyadic = torch.ones(1, *shape_dyadic, 3, dtype=torch.float64) * 0.5
        meta_dyadic = {'shape': shape_dyadic, 'spacing': spacing_dyadic, 'direction': direction_dyadic}

        leg_dyadic = legacy_calc(disp_dyadic, shape_dyadic, meta_dyadic, 3)
        new_dyadic = sp.normalized_to_physical_disp(disp_dyadic, shape_dyadic, spacing_dyadic, direction_dyadic, dtype=torch.float64)
        diff_dyadic = torch.max(torch.abs(leg_dyadic - new_dyadic)).item()
        assert diff_dyadic == 0.0, f"Dyadic L_inf difference is non-zero: {diff_dyadic}"

        # 2. Anisotropic rotated 3D volume in float64
        shape_3d = (12, 16, 20)
        spacing_3d = (0.7, 1.2, 2.5)
        R = Rotation.from_euler('xyz', [25, 40, 65], degrees=True).as_matrix()
        meta_3d = {'shape': shape_3d, 'spacing': spacing_3d, 'direction': R}
        disp_3d = torch.randn(1, *shape_3d, 3, dtype=torch.float64)

        leg_3d = legacy_calc(disp_3d, shape_3d, meta_3d, 3)
        new_3d = sp.normalized_to_physical_disp(disp_3d, shape_3d, spacing_3d, R, dtype=torch.float64)
        diff_3d = torch.max(torch.abs(leg_3d - new_3d)).item()
        assert diff_3d < 1e-12, f"Float64 L_inf difference too large: {diff_3d}"

    def test_to_displacement_field_and_jacobian_anisotropic(self):
        """Task 2 Verification: to_displacement_field() and jacobian_determinant_image() on anisotropic geometry."""
        shape_zyx = (14, 18, 22)
        shape_xyz = (22, 18, 14)
        spacing_xyz = (0.7, 1.3, 2.1)
        origin_xyz = (-25.0, 10.5, 42.0)
        R = Rotation.from_euler('xyz', [25, -35, 50], degrees=True).as_matrix().astype(np.float32)

        meta = {
            'origin': origin_xyz,
            'spacing': spacing_xyz,
            'direction': R,
            'shape': shape_zyx
        }

        # Known physical translation vector
        t_phys = np.array([5.0, -3.0, 2.0], dtype=np.float32)
        u_vox = (R.T @ t_phys) / np.array(spacing_xyz, dtype=np.float32)
        u_norm_xyz = 2.0 * u_vox / (np.array(shape_xyz, dtype=np.float32) - 1.0)

        warp = torch.zeros(1, *shape_zyx, 3)
        warp[..., 0] = float(u_norm_xyz[0])
        warp[..., 1] = float(u_norm_xyz[1])
        warp[..., 2] = float(u_norm_xyz[2])

        tx = SyNToTransform(warp_field=warp, metadata=meta)

        disp_field = tx.to_displacement_field()
        assert isinstance(disp_field, ants.ANTsImage)
        assert disp_field.shape == shape_xyz
        assert np.allclose(disp_field.spacing, spacing_xyz)
        assert np.allclose(disp_field.origin, origin_xyz)
        assert np.allclose(disp_field.direction, R)
        assert disp_field.components == 3

        mean_disp = np.mean(disp_field.numpy(), axis=(0, 1, 2))
        assert np.allclose(mean_disp, t_phys, atol=1e-4)

        jac_img = tx.jacobian_determinant_image()
        assert isinstance(jac_img, ants.ANTsImage)
        assert jac_img.shape == shape_xyz
        assert np.allclose(jac_img.spacing, spacing_xyz)
        assert np.allclose(jac_img.origin, origin_xyz)
        assert np.allclose(jac_img.direction, R)
        # Translation has unit Jacobian
        assert np.allclose(jac_img.numpy(), 1.0, atol=1e-3)
