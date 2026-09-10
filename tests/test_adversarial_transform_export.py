"""
Adversarial stress tests for SyNToTransform.export, to_ants, and serialization.
"""

import os
import pytest
import numpy as np
import torch
import ants

from syntx.transform import SyNToTransform, create_ants_affine
from syntx.spatial import (
    get_identity_grid_torch,
    get_image_metadata,
    grid_to_physical_affine_torch,
    export_ants_affine_transform,
    export_ants_displacement_field,
)


@pytest.fixture
def sample_metadata_3d():
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [10, -15, 25], degrees=True).as_matrix()
    return {
        'origin': (10.5, -20.0, 35.2),
        'spacing': (0.8, 1.2, 1.5),
        'direction': R,
        'shape': (16, 20, 24)
    }


@pytest.fixture
def sample_metadata_2d():
    return {
        'origin': (-5.0, 15.0),
        'spacing': (1.2, 0.9),
        'direction': np.array([
            [0.96, -0.28],
            [0.28, 0.96]
        ], dtype=np.float32),
        'shape': (24, 32)
    }


# ==============================================================================
# 1. Disk Export Stress Tests
# ==============================================================================

class TestDiskExportStress:
    """Stress test disk export with arbitrary prefixes, dimensions, and metadata."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_disk_export_valid_itk_objects(self, dim, tmp_path, sample_metadata_2d, sample_metadata_3d):
        meta = sample_metadata_2d if dim == 2 else sample_metadata_3d
        shape = meta['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        
        # Non-trivial smooth warp
        warp = torch.zeros_like(identity)
        warp[..., 0] = 0.02
        warp[..., 1] = -0.01
        if dim == 3:
            warp[..., 2] = 0.015

        # Non-trivial affine
        M = np.eye(dim, dtype=np.float32)
        M[0, 1] = 0.05
        t = np.linspace(1.0, 3.0, dim, dtype=np.float32)

        tx = SyNToTransform(
            warp_field=warp,
            metadata=meta,
            device='cpu',
            affine_matrix=(M, t)
        )

        deep_prefix = str(tmp_path / "deep" / "nested" / "dir" / "test_out_")
        res = tx.export(outprefix=deep_prefix)

        # Verify returned dictionary structure
        assert 'fwdtransforms' in res
        assert 'invtransforms' in res
        assert 'fwd_transforms' in res
        assert 'inv_transforms' in res
        assert 'affine' in res
        assert 'warp' in res
        assert 'inverse_warp' in res

        affine_path = res['affine']
        warp_path = res['warp']
        inv_warp_path = res['inverse_warp']

        assert affine_path.endswith("0GenericAffine.mat")
        assert warp_path.endswith("1Warp.nii.gz")
        assert inv_warp_path.endswith("1InverseWarp.nii.gz")

        assert os.path.isfile(affine_path)
        assert os.path.isfile(warp_path)
        assert os.path.isfile(inv_warp_path)

        # Verify ITK transform reloadable
        itk_tx = ants.read_transform(affine_path)
        assert isinstance(itk_tx, ants.ANTsTransform)
        assert itk_tx.dimension == dim

        # Verify ITK warp images reloadable
        itk_warp = ants.image_read(warp_path)
        itk_inv_warp = ants.image_read(inv_warp_path)

        assert isinstance(itk_warp, ants.ANTsImage)
        assert isinstance(itk_inv_warp, ants.ANTsImage)
        assert itk_warp.components == dim
        assert itk_inv_warp.components == dim

        # Verify shape in ITK order (reversed from tensor spatial shape)
        itk_shape = tuple(reversed(shape))
        assert itk_warp.shape == itk_shape
        assert itk_inv_warp.shape == itk_shape

        # Verify spacing, origin, direction preserved
        np.testing.assert_allclose(itk_warp.spacing, meta['spacing'], atol=1e-5)
        np.testing.assert_allclose(itk_warp.origin, meta['origin'], atol=1e-5)
        np.testing.assert_allclose(itk_warp.direction, meta['direction'], atol=1e-5)

    def test_disk_export_without_affine(self, tmp_path, sample_metadata_3d):
        """Exporting when there is no affine should omit 0GenericAffine.mat."""
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=None,
            T_grid=None
        )

        prefix = str(tmp_path / "no_aff_")
        res = tx.export(outprefix=prefix)

        assert res['affine'] is None
        assert len(res['fwd_transforms']) == 1
        assert res['fwd_transforms'][0] == res['warp']
        assert len(res['inv_transforms']) == 1
        assert res['inv_transforms'][0] == res['inverse_warp']
        assert not os.path.exists(f"{prefix}0GenericAffine.mat")
        assert os.path.exists(res['warp'])
        assert os.path.exists(res['inverse_warp'])

    def test_disk_export_with_explicit_inv_warp(self, tmp_path, sample_metadata_3d):
        """Exporting when warp_inv_field is explicitly provided."""
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        warp[..., 0] = 0.05
        inv_warp = torch.zeros_like(identity)
        inv_warp[..., 0] = -0.05

        tx = SyNToTransform(
            warp_field=warp,
            warp_inv_field=inv_warp,
            metadata=sample_metadata_3d,
            device='cpu'
        )

        prefix = str(tmp_path / "explicit_inv_")
        res = tx.export(outprefix=prefix)
        inv_read = ants.image_read(res['inverse_warp'])
        assert inv_read.components == 3


# ==============================================================================
# 2. In-Memory Export Tests
# ==============================================================================

class TestInMemoryExport:
    """Test export(outprefix=None) in-memory contract."""

    def test_in_memory_export_keys_and_references(self, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.randn_like(identity) * 0.01
        inv_warp = -warp
        aff_grid = identity.clone()
        M = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)

        tx = SyNToTransform(
            affine_grid=aff_grid,
            warp_field=warp,
            warp_inv_field=inv_warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=(M, t)
        )

        out = tx.export(outprefix=None)
        assert isinstance(out, dict)

        expected_keys = {
            'affine_grid', 'warp_field', 'warp_inv_field',
            'fwd_warp', 'inv_warp', 'affine_matrix', 'metadata'
        }
        assert expected_keys.issubset(set(out.keys()))

        assert out['affine_grid'] is aff_grid
        assert out['warp_field'] is warp
        assert out['fwd_warp'] is warp
        assert out['warp_inv_field'] is inv_warp
        assert out['inv_warp'] is inv_warp
        assert out['metadata'] == sample_metadata_3d
        assert out['affine_matrix'] == (M, t)


# ==============================================================================
# 3. to_ants() Contract Tests
# ==============================================================================

class TestToAntsContract:
    """Test to_ants() with and without outprefix."""

    def test_to_ants_in_memory_with_affine_and_inv_warp(self, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        inv_warp = torch.zeros_like(identity)
        M = np.eye(3, dtype=np.float32)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        tx = SyNToTransform(
            warp_field=warp,
            warp_inv_field=inv_warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=(M, t)
        )

        res = tx.to_ants()
        assert isinstance(res, dict)
        assert 'warp' in res
        assert 'affine' in res
        assert 'inverse_warp' in res

        assert isinstance(res['warp'], ants.ANTsImage)
        assert isinstance(res['affine'], ants.ANTsTransform)
        assert isinstance(res['inverse_warp'], ants.ANTsImage)

    def test_to_ants_in_memory_without_affine(self, sample_metadata_2d):
        shape = sample_metadata_2d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_2d,
            device='cpu'
        )

        res = tx.to_ants()
        assert isinstance(res, dict)
        assert isinstance(res['warp'], ants.ANTsImage)
        assert res['affine'] is None
        # warp_inv_field was not provided
        assert 'inverse_warp' not in res

    def test_to_ants_with_outprefix_delegates_to_export(self, tmp_path, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        M = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=(M, t)
        )

        prefix = str(tmp_path / "to_ants_prefix_")
        res = tx.to_ants(outprefix=prefix)
        assert isinstance(res, dict)
        assert 'fwdtransforms' in res
        assert os.path.exists(res['affine'])
        assert os.path.exists(res['warp'])
        assert os.path.exists(res['inverse_warp'])


# ==============================================================================
# 4. Batched vs Unbatched T_grid Export Tests
# ==============================================================================

class TestTGridBatching:
    """Stress test batched (1, dim, dim+1) and unbatched (dim, dim+1) T_grid."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_t_grid_batched_vs_unbatched_export_parity(self, dim, tmp_path, sample_metadata_2d, sample_metadata_3d):
        meta = sample_metadata_2d if dim == 2 else sample_metadata_3d
        shape = meta['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)

        # Construct non-trivial T_grid: scale by 1.1 and translate by 0.05
        T_unbatched = torch.eye(dim + 1)[:dim, :].clone()
        T_unbatched[0, 0] = 1.1
        T_unbatched[0, dim] = 0.05
        T_batched = T_unbatched.unsqueeze(0)

        assert T_unbatched.shape == (dim, dim + 1)
        assert T_batched.shape == (1, dim, dim + 1)

        tx_unbatched = SyNToTransform(
            warp_field=warp,
            metadata=meta,
            device='cpu',
            T_grid=T_unbatched
        )

        tx_batched = SyNToTransform(
            warp_field=warp,
            metadata=meta,
            device='cpu',
            T_grid=T_batched
        )

        res_unbatched = tx_unbatched.export(outprefix=str(tmp_path / "unbatched_"))
        res_batched = tx_batched.export(outprefix=str(tmp_path / "batched_"))

        assert os.path.isfile(res_unbatched['affine'])
        assert os.path.isfile(res_batched['affine'])

        tx1 = ants.read_transform(res_unbatched['affine'])
        tx2 = ants.read_transform(res_batched['affine'])

        # Check transform parameters match exactly
        np.testing.assert_allclose(tx1.parameters, tx2.parameters, atol=1e-5)
        np.testing.assert_allclose(tx1.fixed_parameters, tx2.fixed_parameters, atol=1e-5)


# ==============================================================================
# 5. Affine Matrix Format Variations
# ==============================================================================

class TestAffineFormatVariations:
    """Stress test different inputs for affine_matrix."""

    def test_affine_as_ants_transform(self, tmp_path, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        ants_tx = create_ants_affine(np.eye(3), np.array([2.0, -1.0, 0.5]), dim=3)

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=ants_tx
        )
        res = tx.export(outprefix=str(tmp_path / "ants_tx_"))
        reloaded = ants.read_transform(res['affine'])
        np.testing.assert_allclose(reloaded.parameters, ants_tx.parameters, atol=1e-5)

    def test_affine_as_homogeneous_square_matrix(self, tmp_path, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        homo = np.eye(4, dtype=np.float32)
        homo[0, 3] = 4.2
        homo[1, 3] = -2.1

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=homo
        )
        res = tx.export(outprefix=str(tmp_path / "homo_3d_"))
        reloaded = ants.read_transform(res['affine'])
        assert isinstance(reloaded, ants.ANTsTransform)

    def test_affine_as_dim_by_dim_plus_one(self, tmp_path, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        aug = np.eye(4, dtype=np.float32)[:3, :]
        aug[0, 3] = 1.5

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=aug
        )
        res = tx.export(outprefix=str(tmp_path / "aug_3d_"))
        reloaded = ants.read_transform(res['affine'])
        assert isinstance(reloaded, ants.ANTsTransform)

    def test_invalid_affine_shape_raises_value_error(self, tmp_path, sample_metadata_3d):
        shape = sample_metadata_3d['shape']
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        invalid_mat = np.eye(2, dtype=np.float32)  # 2x2 for 3D

        tx = SyNToTransform(
            warp_field=warp,
            metadata=sample_metadata_3d,
            device='cpu',
            affine_matrix=invalid_mat
        )
        with pytest.raises(ValueError, match="Unsupported affine_matrix shape"):
            tx.export(outprefix=str(tmp_path / "invalid_"))


# ==============================================================================
# 6. ANTs Apply Transforms Roundtrip Interoperability
# ==============================================================================

class TestAntsApplyTransformsInteroperability:
    """Verify that exported files are valid for ants.apply_transforms."""

    def test_ants_apply_transforms_runs_cleanly(self, tmp_path):
        # Create synthetic ANTsImages
        shape = (16, 16, 16)
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)

        fixed_np = np.zeros(shape, dtype=np.float32)
        fixed_np[4:12, 4:12, 4:12] = 1.0
        fixed = ants.from_numpy(fixed_np, origin=origin, spacing=spacing, direction=direction)

        moving = fixed.clone()

        # Build transform with known small shift
        identity = get_identity_grid_torch(shape, device='cpu')
        warp = torch.zeros_like(identity)
        M = np.eye(3, dtype=np.float32)
        t = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        meta = get_image_metadata(fixed)
        tx = SyNToTransform(
            warp_field=warp,
            metadata=meta,
            device='cpu',
            affine_matrix=(M, t)
        )

        res = tx.export(outprefix=str(tmp_path / "interop_"))
        fwd_list = res['fwdtransforms']

        # Apply transforms in ANTsPy
        warped_ants = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=fwd_list,
            interpolator='linear'
        )

        assert isinstance(warped_ants, ants.ANTsImage)
        assert warped_ants.shape == fixed.shape
        assert not np.isnan(warped_ants.numpy()).any()


# ==============================================================================
# 7. Empirical Failure Modes (Adversarial Bug Disclosures)
# ==============================================================================

class TestEmpiricalFailureModes:
    """Empirical demonstrations of confirmed failure modes in SyNToTransform."""

    def test_failure_mode_1_tgrid_axis_alignment(self, tmp_path):
        """Translation along X in T_grid should produce translation along X in ITK transform."""
        arr = np.zeros((30, 40, 50), dtype=np.float32)
        fixed = ants.from_numpy(arr, spacing=(1.0, 2.0, 3.0), origin=(0.0, 0.0, 0.0))
        meta = get_image_metadata(fixed)

        # Translation in normalized X
        T_grid = torch.eye(4)[:3, :]
        T_grid[0, 3] = 0.5

        warp = torch.zeros(1, 50, 40, 30, 3)
        tx = SyNToTransform(warp_field=warp, metadata=meta, T_grid=T_grid)
        res = tx.export(outprefix=str(tmp_path / "fm1_"))

        tx_itk = ants.read_transform(res['affine'])
        # ITK parameters: [M_00..M_22, t_x, t_y, t_z]
        t_x = tx_itk.parameters[9]
        t_y = tx_itk.parameters[10]
        t_z = tx_itk.parameters[11]

        # Expected: non-zero shift in X, zero shift in Y and Z
        assert abs(t_x) > 1.0, f"Expected non-zero t_x, got {t_x}"
        assert abs(t_z) < 1e-4, f"Expected zero t_z, got {t_z} (X and Z axes inverted!)"

    def test_failure_mode_2_to_ants_drops_tgrid(self):
        """to_ants(outprefix=None) should return an ANTsTransform when T_grid is provided."""
        arr = np.zeros((16, 16), dtype=np.float32)
        fixed = ants.from_numpy(arr, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        meta = get_image_metadata(fixed)

        T_grid = torch.eye(3)[:2, :].unsqueeze(0)
        warp = torch.zeros(1, 16, 16, 2)
        tx = SyNToTransform(warp_field=warp, metadata=meta, T_grid=T_grid)

        res = tx.to_ants()
        assert res['affine'] is not None, "to_ants() dropped T_grid; res['affine'] is None"
        assert isinstance(res['affine'], ants.ANTsTransform)

    def test_failure_mode_3_invert_drops_tgrid(self):
        """invert() should preserve and invert T_grid."""
        arr = np.zeros((16, 16), dtype=np.float32)
        fixed = ants.from_numpy(arr, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        meta = get_image_metadata(fixed)

        T_grid = torch.eye(3)[:2, :].unsqueeze(0)
        warp = torch.zeros(1, 16, 16, 2)
        tx = SyNToTransform(warp_field=warp, metadata=meta, T_grid=T_grid)

        inv_tx = tx.invert()
        has_affine = inv_tx.affine_matrix is not None or getattr(inv_tx, 'T_grid', None) is not None
        assert has_affine, "invert() dropped T_grid; both affine_matrix and T_grid are None"

    def test_failure_mode_4_non_cubic_3d_target_shape_mismatch(self):
        """tx.apply() should not alter spatial dimensions for non-cubic 3D images."""
        arr = np.zeros((16, 24, 32), dtype=np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        meta = get_image_metadata(img)

        # Spatial shape in tensor order (Z, Y, X) is (32, 24, 16)
        tensor_spatial = (32, 24, 16)
        warp = torch.zeros(1, *tensor_spatial, 3)
        tx = SyNToTransform(warp_field=warp, metadata=meta)

        img_tensor = torch.zeros(1, 1, *tensor_spatial)
        warped = tx.apply(img_tensor)

        # Output shape should remain (1, 1, 32, 24, 16), not swapped to (1, 1, 16, 24, 32)
        assert warped.shape == img_tensor.shape, f"Shape mutated: expected {img_tensor.shape}, got {warped.shape}"


# ==============================================================================
# 8. Affine-Only Transformation & Export Regression Suite
# ==============================================================================

class TestAffineOnlyTransformRegressions:
    """Regression test suite guarding against affine-only transform failure modes
    identified during adversarial verification of Milestone 2 (challenger_m2_gate2_1)."""

    def test_affine_only_non_cubic_3d_apply_no_dimension_swap(self):
        """Verify that tx = SyNToTransform(metadata=meta, T_grid=T_grid) on a (36, 22, 14)
        volume preserves input tensor shape (1, 1, 14, 22, 36) without dimension swapping."""
        # ITK image with anisotropic / non-cubic dimensions (X=36, Y=22, Z=14)
        itk_shape = (36, 22, 14)
        arr = np.zeros(itk_shape, dtype=np.float32)
        img = ants.from_numpy(arr, spacing=(0.8, 1.2, 1.5), origin=(10.0, -5.0, 20.0))
        meta = get_image_metadata(img)  # meta['shape'] == (36, 22, 14)

        # PyTorch image tensor is ordered (B, C, Z, Y, X) = (1, 1, 14, 22, 36)
        tensor_spatial = tuple(reversed(itk_shape))  # (14, 22, 36)
        img_tensor = torch.randn(1, 1, *tensor_spatial, dtype=torch.float32)

        # Affine grid transform (identity normalized grid)
        T_grid = torch.eye(4, dtype=torch.float32)[:3, :].unsqueeze(0)

        # SyNToTransform instantiated with NO displacement field (warp_field=None)
        tx = SyNToTransform(metadata=meta, T_grid=T_grid)

        # Verify internal shape attributes are in PyTorch tensor order (Z, Y, X)
        assert tx.spatial == tensor_spatial, (
            f"tx.spatial must be in PyTorch order {tensor_spatial}, got {tx.spatial}"
        )
        assert tx.target_shape == tensor_spatial, (
            f"tx.target_shape must be in PyTorch order {tensor_spatial}, got {tx.target_shape}"
        )

        warped = tx.apply(img_tensor)

        # Output shape must strictly match input shape (1, 1, 14, 22, 36)
        assert warped.shape == img_tensor.shape, (
            f"Dimension swap detected! Expected {img_tensor.shape}, got {warped.shape}"
        )
        assert not torch.isnan(warped).any(), "Resampled tensor contains NaNs"

    def test_affine_only_to_displacement_field_shape(self):
        """Verify that tx.to_displacement_field() has ITK shape (36, 22, 14)
        when tx is an affine-only transform on a non-cubic volume."""
        itk_shape = (36, 22, 14)
        arr = np.zeros(itk_shape, dtype=np.float32)
        spacing = (0.8, 1.2, 1.5)
        origin = (10.0, -5.0, 20.0)
        direction = np.eye(3, dtype=np.float32)
        img = ants.from_numpy(arr, spacing=spacing, origin=origin, direction=direction)
        meta = get_image_metadata(img)

        # Case A: Affine-only via affine_matrix (physical affine parameters)
        M = np.eye(3, dtype=np.float32)
        t = np.array([2.0, -1.0, 3.0], dtype=np.float32)
        tx_aff = SyNToTransform(metadata=meta, affine_matrix=(M, t))
        disp_aff = tx_aff.to_displacement_field()

        assert isinstance(disp_aff, ants.ANTsImage), "Output must be an ants.ANTsImage"
        assert disp_aff.shape == itk_shape, (
            f"ANTsImage shape inverted! Expected ITK shape {itk_shape}, got {disp_aff.shape}"
        )
        assert disp_aff.components == 3, f"Expected 3 components, got {disp_aff.components}"
        np.testing.assert_allclose(disp_aff.spacing, spacing, atol=1e-5)
        np.testing.assert_allclose(disp_aff.origin, origin, atol=1e-5)
        np.testing.assert_allclose(disp_aff.direction, direction, atol=1e-5)

        # Case B: Affine-only via T_grid (normalized grid affine)
        T_grid = torch.eye(4, dtype=torch.float32)[:3, :].unsqueeze(0)
        tx_tgrid = SyNToTransform(metadata=meta, T_grid=T_grid)
        disp_tgrid = tx_tgrid.to_displacement_field()

        assert isinstance(disp_tgrid, ants.ANTsImage)
        assert disp_tgrid.shape == itk_shape, (
            f"ANTsImage shape inverted for T_grid! Expected {itk_shape}, got {disp_tgrid.shape}"
        )
        assert disp_tgrid.components == 3

    def test_affine_only_to_ants_no_crash(self):
        """Verify that tx.to_ants() returns {'warp': None, 'affine': ANTsTransform}
        without raising TypeError when warp_field is None."""
        # 2D case
        arr_2d = np.zeros((16, 20), dtype=np.float32)
        img_2d = ants.from_numpy(arr_2d, spacing=(1.0, 1.2), origin=(0.0, 5.0))
        meta_2d = get_image_metadata(img_2d)

        M_2d = np.eye(2, dtype=np.float32)
        t_2d = np.array([1.5, -2.0], dtype=np.float32)
        tx_2d = SyNToTransform(metadata=meta_2d, affine_matrix=(M_2d, t_2d))

        res_2d = tx_2d.to_ants()
        assert isinstance(res_2d, dict), "to_ants() must return a dictionary"
        assert res_2d.get('warp') is None, f"Expected res['warp'] is None, got {res_2d.get('warp')}"
        assert res_2d.get('affine') is not None, "Expected res['affine'] to be present"
        assert isinstance(res_2d['affine'], ants.ANTsTransform), (
            f"res['affine'] must be ANTsTransform, got {type(res_2d['affine'])}"
        )
        assert res_2d['affine'].dimension == 2

        # 3D case with T_grid
        arr_3d = np.zeros((36, 22, 14), dtype=np.float32)
        img_3d = ants.from_numpy(arr_3d, spacing=(0.8, 1.2, 1.5), origin=(10.0, -5.0, 20.0))
        meta_3d = get_image_metadata(img_3d)

        T_grid = torch.eye(4, dtype=torch.float32)[:3, :].unsqueeze(0)
        tx_3d = SyNToTransform(metadata=meta_3d, T_grid=T_grid)

        res_3d = tx_3d.to_ants()
        assert isinstance(res_3d, dict)
        assert res_3d.get('warp') is None
        assert res_3d.get('affine') is not None
        assert isinstance(res_3d['affine'], ants.ANTsTransform)
        assert res_3d['affine'].dimension == 3

    def test_affine_only_disk_export_no_crash(self, tmp_path):
        """Verify that tx.export(outprefix=...) writes 0GenericAffine.mat and returns
        fwdtransforms = [affine_path] without attempting to write 1Warp.nii.gz from None."""
        # 3D non-cubic case
        itk_shape = (36, 22, 14)
        arr = np.zeros(itk_shape, dtype=np.float32)
        spacing = (0.8, 1.2, 1.5)
        origin = (10.0, -5.0, 20.0)
        img = ants.from_numpy(arr, spacing=spacing, origin=origin)
        meta = get_image_metadata(img)

        M = np.eye(3, dtype=np.float32)
        t = np.array([2.0, -1.0, 3.0], dtype=np.float32)
        tx = SyNToTransform(metadata=meta, affine_matrix=(M, t))

        prefix = str(tmp_path / "affine_only_export_")
        res = tx.export(outprefix=prefix)

        # 1. Verify dictionary structure and keys
        assert isinstance(res, dict)
        assert 'fwdtransforms' in res
        assert 'invtransforms' in res
        assert 'fwd_transforms' in res
        assert 'inv_transforms' in res
        assert 'affine' in res
        assert 'warp' in res
        assert 'inverse_warp' in res

        affine_path = res['affine']
        assert affine_path is not None
        assert affine_path.endswith("0GenericAffine.mat")
        assert os.path.isfile(affine_path), f"Affine file not found: {affine_path}"

        # 2. Warp entries must be None and warp files must NOT be written to disk
        assert res['warp'] is None, f"Expected res['warp'] is None, got {res['warp']}"
        assert res['inverse_warp'] is None, f"Expected res['inverse_warp'] is None, got {res['inverse_warp']}"
        assert not os.path.exists(f"{prefix}1Warp.nii.gz"), "1Warp.nii.gz should not be written for affine-only"
        assert not os.path.exists(f"{prefix}1InverseWarp.nii.gz"), "1InverseWarp.nii.gz should not be written for affine-only"

        # 3. Transform lists must only contain the affine transform
        assert res['fwdtransforms'] == [affine_path]
        assert res['fwd_transforms'] == [affine_path]
        assert res['invtransforms'] == [affine_path]
        assert res['inv_transforms'] == [affine_path]

        # 4. ITK transform must be readable and valid
        itk_tx = ants.read_transform(affine_path)
        assert isinstance(itk_tx, ants.ANTsTransform)
        assert itk_tx.dimension == 3

        # 5. Must also succeed when initialized via T_grid
        prefix_tgrid = str(tmp_path / "tgrid_only_export_")
        T_grid = torch.eye(4, dtype=torch.float32)[:3, :].unsqueeze(0)
        tx_tgrid = SyNToTransform(metadata=meta, T_grid=T_grid)
        res_tgrid = tx_tgrid.export(outprefix=prefix_tgrid)

        assert res_tgrid['warp'] is None
        assert res_tgrid['inverse_warp'] is None
        assert res_tgrid['affine'] is not None
        assert os.path.isfile(res_tgrid['affine'])
        assert res_tgrid['fwdtransforms'] == [res_tgrid['affine']]
        assert not os.path.exists(f"{prefix_tgrid}1Warp.nii.gz")

        # 6. Verify interoperability with ants.apply_transforms
        moving = img.clone()
        warped_ants = ants.apply_transforms(
            fixed=img,
            moving=moving,
            transformlist=res['fwdtransforms'],
            interpolator='linear'
        )
        assert isinstance(warped_ants, ants.ANTsImage)
        assert warped_ants.shape == itk_shape

    def test_affine_only_2d_non_square_anisotropic_rotated_apply(self):
        """Verify that tx.apply() on a 2D non-square image (48, 20) with anisotropic spacing
        and non-trivial rotation/translation preserves input tensor shape (1, 1, 20, 48)."""
        shape_2d_itk = (48, 20)
        spacing_2d = (0.4, 1.8)
        origin_2d = (-5.0, 15.0)
        theta = float(np.deg2rad(25.0))
        R_2d = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=np.float32)
        arr_2d = np.zeros(shape_2d_itk, dtype=np.float32)
        img_2d = ants.from_numpy(arr_2d, spacing=spacing_2d, origin=origin_2d, direction=R_2d)
        meta_2d = get_image_metadata(img_2d)

        phi = float(np.deg2rad(-15.0))
        M_2d = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ], dtype=np.float32)
        t_2d = np.array([2.5, -3.8], dtype=np.float32)

        tensor_shape_2d = (20, 48)  # (H, W) = (Ny, Nx)
        img_t_2d = torch.randn(1, 1, *tensor_shape_2d, dtype=torch.float32)

        # Test both affine_matrix and T_grid, both is_physical=False and is_physical=True
        for is_phys in [False, True]:
            tx_aff = SyNToTransform(metadata=meta_2d, affine_matrix=(M_2d, t_2d), is_physical=is_phys)
            assert tx_aff.spatial == tensor_shape_2d
            assert tx_aff.target_shape == tensor_shape_2d
            warped = tx_aff.apply(img_t_2d)
            assert warped.shape == (1, 1, 20, 48), f"2D shape mismatch: {warped.shape}"
            assert not torch.isnan(warped).any()

            T_2d = torch.eye(3, dtype=torch.float32)[:2, :].unsqueeze(0)
            T_2d[0, 0, 2] = 0.2
            T_2d[0, 1, 2] = -0.15
            tx_tgrid = SyNToTransform(metadata=meta_2d, T_grid=T_2d, is_physical=is_phys)
            warped_tgrid = tx_tgrid.apply(img_t_2d)
            assert warped_tgrid.shape == (1, 1, 20, 48), f"2D T_grid shape mismatch: {warped_tgrid.shape}"
            assert not torch.isnan(warped_tgrid).any()

    def test_affine_only_3d_non_cubic_anisotropic_rotated_apply(self):
        """Verify that tx.apply() on a 3D non-cubic volume (36, 22, 14) with anisotropic spacing
        and non-trivial 3D rotation/translation preserves input tensor shape (1, 1, 14, 22, 36)."""
        shape_3d_itk = (36, 22, 14)
        spacing_3d = (0.5, 1.2, 2.5)
        origin_3d = (10.0, -20.0, 30.0)
        from scipy.spatial.transform import Rotation
        R_3d = Rotation.from_euler('xyz', [15, -20, 30], degrees=True).as_matrix().astype(np.float32)
        arr_3d = np.zeros(shape_3d_itk, dtype=np.float32)
        img_3d = ants.from_numpy(arr_3d, spacing=spacing_3d, origin=origin_3d, direction=R_3d)
        meta_3d = get_image_metadata(img_3d)

        M_3d = Rotation.from_euler('xyz', [10, 5, -15], degrees=True).as_matrix().astype(np.float32)
        t_3d = np.array([3.5, -2.1, 4.2], dtype=np.float32)

        tensor_shape_3d = (14, 22, 36)  # (D, H, W) = (Nz, Ny, Nx)
        img_t_3d = torch.randn(1, 1, *tensor_shape_3d, dtype=torch.float32)

        for is_phys in [False, True]:
            tx_aff = SyNToTransform(metadata=meta_3d, affine_matrix=(M_3d, t_3d), is_physical=is_phys)
            assert tx_aff.spatial == tensor_shape_3d
            assert tx_aff.target_shape == tensor_shape_3d
            warped = tx_aff.apply(img_t_3d)
            assert warped.shape == (1, 1, 14, 22, 36), f"3D shape mismatch: {warped.shape}"
            assert not torch.isnan(warped).any()

            T_3d = torch.eye(4, dtype=torch.float32)[:3, :].unsqueeze(0)
            T_3d[0, 0, 3] = 0.25
            T_3d[0, 1, 3] = -0.1
            T_3d[0, 2, 3] = 0.15
            tx_tgrid = SyNToTransform(metadata=meta_3d, T_grid=T_3d, is_physical=is_phys)
            warped_tgrid = tx_tgrid.apply(img_t_3d)
            assert warped_tgrid.shape == (1, 1, 14, 22, 36), f"3D T_grid shape mismatch: {warped_tgrid.shape}"
            assert not torch.isnan(warped_tgrid).any()

    def test_affine_only_to_displacement_field_and_jacobian_2d_and_3d(self):
        """Verify to_displacement_field() and jacobian() for affine-only transforms across 2D/3D."""
        from scipy.spatial.transform import Rotation

        # 2D non-square
        shape_2d = (48, 20)
        img_2d = ants.from_numpy(np.zeros(shape_2d, dtype=np.float32), spacing=(0.4, 1.8), origin=(-5.0, 15.0))
        meta_2d = get_image_metadata(img_2d)
        phi = float(np.deg2rad(-15.0))
        M_2d = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]], dtype=np.float32)
        t_2d = np.array([2.5, -3.8], dtype=np.float32)

        for is_phys in [False, True]:
            tx_2d = SyNToTransform(metadata=meta_2d, affine_matrix=(M_2d, t_2d), is_physical=is_phys)
            disp_2d = tx_2d.to_displacement_field()
            assert isinstance(disp_2d, ants.ANTsImage)
            assert disp_2d.shape == shape_2d
            assert disp_2d.components == 2

            jac_2d = tx_2d.jacobian()
            assert jac_2d.shape == (20, 48)
            assert np.allclose(jac_2d, 1.0, atol=2e-2)

        # 3D non-cubic
        shape_3d = (36, 22, 14)
        R_3d = Rotation.from_euler('xyz', [15, -20, 30], degrees=True).as_matrix().astype(np.float32)
        img_3d = ants.from_numpy(np.zeros(shape_3d, dtype=np.float32), spacing=(0.5, 1.2, 2.5), origin=(10.0, -20.0, 30.0), direction=R_3d)
        meta_3d = get_image_metadata(img_3d)
        M_3d = Rotation.from_euler('xyz', [10, 5, -15], degrees=True).as_matrix().astype(np.float32)
        t_3d = np.array([3.5, -2.1, 4.2], dtype=np.float32)

        for is_phys in [False, True]:
            tx_3d = SyNToTransform(metadata=meta_3d, affine_matrix=(M_3d, t_3d), is_physical=is_phys)
            disp_3d = tx_3d.to_displacement_field()
            assert isinstance(disp_3d, ants.ANTsImage)
            assert disp_3d.shape == shape_3d
            assert disp_3d.components == 3

            jac_3d = tx_3d.jacobian()
            assert jac_3d.shape == (14, 22, 36)
            assert np.allclose(jac_3d, 1.0, atol=2e-2)

    def test_affine_only_2d_and_3d_to_ants_and_disk_export_reload_cleanly(self, tmp_path):
        """Verify to_ants() and disk export() across 2D/3D non-square/non-cubic geometries with reloading."""
        from scipy.spatial.transform import Rotation

        # 2D non-square
        shape_2d = (48, 20)
        img_2d = ants.from_numpy(np.zeros(shape_2d, dtype=np.float32), spacing=(0.4, 1.8), origin=(-5.0, 15.0))
        meta_2d = get_image_metadata(img_2d)
        phi = float(np.deg2rad(-15.0))
        M_2d = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]], dtype=np.float32)
        t_2d = np.array([2.5, -3.8], dtype=np.float32)

        tx_2d = SyNToTransform(metadata=meta_2d, affine_matrix=(M_2d, t_2d))
        res_ants_2d = tx_2d.to_ants()
        assert res_ants_2d['warp'] is None
        assert res_ants_2d['inverse_warp'] is None
        assert isinstance(res_ants_2d['affine'], ants.ANTsTransform)
        assert res_ants_2d['affine'].dimension == 2

        prefix_2d = str(tmp_path / "exp_2d_")
        res_exp_2d = tx_2d.export(outprefix=prefix_2d)
        assert res_exp_2d['warp'] is None
        assert res_exp_2d['inverse_warp'] is None
        assert not os.path.exists(f"{prefix_2d}1Warp.nii.gz")
        assert not os.path.exists(f"{prefix_2d}1InverseWarp.nii.gz")
        assert res_exp_2d['fwdtransforms'] == [res_exp_2d['affine']]

        reloaded_2d = ants.read_transform(res_exp_2d['affine'])
        assert reloaded_2d.dimension == 2
        warped_ants_2d = ants.apply_transforms(fixed=img_2d, moving=img_2d, transformlist=res_exp_2d['fwdtransforms'])
        assert warped_ants_2d.shape == shape_2d

        # 3D non-cubic
        shape_3d = (36, 22, 14)
        R_3d = Rotation.from_euler('xyz', [15, -20, 30], degrees=True).as_matrix().astype(np.float32)
        img_3d = ants.from_numpy(np.zeros(shape_3d, dtype=np.float32), spacing=(0.5, 1.2, 2.5), origin=(10.0, -20.0, 30.0), direction=R_3d)
        meta_3d = get_image_metadata(img_3d)
        M_3d = Rotation.from_euler('xyz', [10, 5, -15], degrees=True).as_matrix().astype(np.float32)
        t_3d = np.array([3.5, -2.1, 4.2], dtype=np.float32)

        tx_3d = SyNToTransform(metadata=meta_3d, affine_matrix=(M_3d, t_3d))
        res_ants_3d = tx_3d.to_ants()
        assert res_ants_3d['warp'] is None
        assert res_ants_3d['inverse_warp'] is None
        assert isinstance(res_ants_3d['affine'], ants.ANTsTransform)
        assert res_ants_3d['affine'].dimension == 3

        prefix_3d = str(tmp_path / "exp_3d_")
        res_exp_3d = tx_3d.export(outprefix=prefix_3d)
        assert res_exp_3d['warp'] is None
        assert res_exp_3d['inverse_warp'] is None
        assert not os.path.exists(f"{prefix_3d}1Warp.nii.gz")
        assert not os.path.exists(f"{prefix_3d}1InverseWarp.nii.gz")
        assert res_exp_3d['fwdtransforms'] == [res_exp_3d['affine']]

        reloaded_3d = ants.read_transform(res_exp_3d['affine'])
        assert reloaded_3d.dimension == 3
        warped_ants_3d = ants.apply_transforms(fixed=img_3d, moving=img_3d, transformlist=res_exp_3d['fwdtransforms'])
        assert warped_ants_3d.shape == shape_3d

    def test_affine_only_ants_apply_numerical_parity_2d_and_3d(self, tmp_path):
        """Verify numerical parity (<0.05 max diff) between tx.apply() and ants.apply_transforms()."""
        # 2D case
        shape_2d = (32, 48)  # X=32, Y=48
        spacing_2d = (0.5, 1.2)
        x = np.linspace(-2, 2, shape_2d[0])
        y = np.linspace(-2, 2, shape_2d[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        blob_2d = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)
        img_2d = ants.from_numpy(blob_2d, spacing=spacing_2d)
        meta_2d = get_image_metadata(img_2d)

        M_2d = np.eye(2, dtype=np.float32)
        t_2d = np.array([1.5, -2.0], dtype=np.float32)
        tx_2d = SyNToTransform(metadata=meta_2d, affine_matrix=(M_2d, t_2d))

        prefix_2d = str(tmp_path / "parity_2d_")
        res_2d = tx_2d.export(outprefix=prefix_2d)

        warped_ants_2d = ants.apply_transforms(
            fixed=img_2d, moving=img_2d,
            transformlist=res_2d['fwdtransforms'],
            interpolator='linear'
        )

        tensor_2d = torch.from_numpy(blob_2d.T).unsqueeze(0).unsqueeze(0)  # (1, 1, Y, X)
        warped_torch_2d = tx_2d.apply(tensor_2d, mode='bilinear')
        arr_warped_2d = warped_torch_2d.squeeze().cpu().numpy().T  # (X, Y)

        diff_2d = np.abs(warped_ants_2d.numpy() - arr_warped_2d)
        assert np.max(diff_2d) < 0.05, f"2D ANTs vs PyTorch diff too large: {np.max(diff_2d)}"

        # 3D case
        shape_3d = (20, 28, 16)  # X=20, Y=28, Z=16
        spacing_3d = (0.8, 1.2, 1.5)
        x = np.linspace(-2, 2, shape_3d[0])
        y = np.linspace(-2, 2, shape_3d[1])
        z = np.linspace(-2, 2, shape_3d[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        blob_3d = np.exp(-(xx**2 + yy**2 + zz**2) / 0.5).astype(np.float32)
        img_3d = ants.from_numpy(blob_3d, spacing=spacing_3d)
        meta_3d = get_image_metadata(img_3d)

        M_3d = np.eye(3, dtype=np.float32)
        t_3d = np.array([2.0, -1.0, 1.5], dtype=np.float32)
        tx_3d = SyNToTransform(metadata=meta_3d, affine_matrix=(M_3d, t_3d))

        prefix_3d = str(tmp_path / "parity_3d_")
        res_3d = tx_3d.export(outprefix=prefix_3d)

        warped_ants_3d = ants.apply_transforms(
            fixed=img_3d, moving=img_3d,
            transformlist=res_3d['fwdtransforms'],
            interpolator='linear'
        )

        tensor_3d = torch.from_numpy(blob_3d.transpose(2, 1, 0)).unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
        warped_torch_3d = tx_3d.apply(tensor_3d, mode='bilinear')
        arr_warped_3d = warped_torch_3d.squeeze().cpu().numpy().transpose(2, 1, 0)  # (X, Y, Z)

        diff_3d = np.abs(warped_ants_3d.numpy() - arr_warped_3d)
        assert np.max(diff_3d) < 0.05, f"3D ANTs vs PyTorch diff too large: {np.max(diff_3d)}"


