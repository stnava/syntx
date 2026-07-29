"""
tests/test_spatial.py — Exhaustive validation of syntx.spatial conversion suite.

Validates all ITK/ANTs ↔ PyTorch/JAX conversions against ANTs C++ ITK reference
for 2D, 3D, and 4D (multi-component) displacement fields.
"""

import pytest
import numpy as np
import torch
import ants
import tempfile
import os

from syntx.spatial import (
    reverse_components,
    reverse_metadata,
    get_image_metadata,
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    image_to_tensor,
    tensor_to_image,
    jacobian_determinant,
    jacobian_determinant_image,
    deformation_stats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def r16_r64_registration():
    """2D SyN registration of r16 ↔ r64 ANTsPy test images."""
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    reg = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
    return fi, mi, reg


@pytest.fixture
def r16_r64_syntx():
    """2D syntx.syn registration of r16 ↔ r64."""
    from syntx import registration
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    fi_norm = (fi - fi.min()) / (fi.max() - fi.min() + 1e-8)
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)
    reg = registration(fixed=fi_norm, moving=mi_norm, type_of_transform='SyN', verbose=False)
    return fi_norm, mi_norm, reg


# ═══════════════════════════════════════════════════════════════════════════════
# Test: reverse_components
# ═══════════════════════════════════════════════════════════════════════════════

class TestReverseComponents:
    def test_2d_numpy(self):
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        rev = reverse_components(arr)
        assert rev.shape == arr.shape
        np.testing.assert_array_equal(rev[0, 0], [2, 1])
        np.testing.assert_array_equal(rev[1, 1], [8, 7])

    def test_3d_numpy(self):
        arr = np.random.randn(4, 5, 6, 3).astype(np.float32)
        rev = reverse_components(arr)
        np.testing.assert_array_equal(rev[..., 0], arr[..., 2])
        np.testing.assert_array_equal(rev[..., 1], arr[..., 1])
        np.testing.assert_array_equal(rev[..., 2], arr[..., 0])

    def test_symmetry(self):
        """Applying reverse_components twice returns the original."""
        arr = np.random.randn(8, 8, 2).astype(np.float32)
        np.testing.assert_array_equal(reverse_components(reverse_components(arr)), arr)

    def test_torch_tensor(self):
        t = torch.randn(1, 8, 8, 2)
        rev = reverse_components(t)
        assert isinstance(rev, np.ndarray)
        np.testing.assert_array_almost_equal(rev[0, ..., 0], t[0, ..., 1].numpy())

    def test_batched(self):
        arr = np.random.randn(2, 8, 8, 3).astype(np.float32)
        rev = reverse_components(arr)
        np.testing.assert_array_equal(rev[0, ..., 0], arr[0, ..., 2])


# ═══════════════════════════════════════════════════════════════════════════════
# Test: reverse_metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestReverseMetadata:
    def test_2d(self):
        sp = (1.0, 2.0)
        og = (10.0, 20.0)
        d = np.eye(2)
        sp_r, og_r, d_r = reverse_metadata(sp, og, d)
        assert sp_r == (2.0, 1.0)
        assert og_r == (20.0, 10.0)
        np.testing.assert_array_equal(d_r, np.eye(2))

    def test_3d(self):
        sp = (1.0, 2.0, 3.0)
        og = (10.0, 20.0, 30.0)
        d = np.diag([1.0, -1.0, 1.0])
        sp_r, og_r, d_r = reverse_metadata(sp, og, d)
        assert sp_r == (3.0, 2.0, 1.0)
        assert og_r == (30.0, 20.0, 10.0)
        expected_d = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
        np.testing.assert_array_equal(d_r, expected_d)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: get_image_metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetImageMetadata:
    def test_basic(self):
        img = ants.image_read(ants.get_data('r16'))
        meta = get_image_metadata(img)
        assert 'origin' in meta
        assert 'spacing' in meta
        assert 'direction' in meta
        assert 'shape' in meta
        assert meta['shape'] == tuple(img.shape)
        assert meta['spacing'] == tuple(img.spacing)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: disp_tensor_to_itk / disp_itk_to_tensor Round-Trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestDispConversions:
    def test_round_trip_2d(self):
        """tensor → ITK → tensor preserves values exactly."""
        fi = ants.image_read(ants.get_data('r16'))
        disp = torch.randn(1, 256, 256, 2)
        itk_img = disp_tensor_to_itk(disp, fi)
        assert itk_img.components == 2
        recovered = disp_itk_to_tensor(itk_img)
        np.testing.assert_array_almost_equal(
            recovered.numpy(), disp.numpy(), decimal=5
        )

    def test_round_trip_3d(self):
        """tensor → ITK → tensor round-trip for 3D fields."""
        fi = ants.from_numpy(
            np.zeros((32, 32, 32), dtype=np.float32),
            spacing=(1.0, 1.0, 1.0)
        )
        disp = torch.randn(1, 32, 32, 32, 3)
        itk_img = disp_tensor_to_itk(disp, fi)
        assert itk_img.components == 3
        recovered = disp_itk_to_tensor(itk_img)
        np.testing.assert_array_almost_equal(
            recovered.numpy(), disp.numpy(), decimal=5
        )

    def test_component_order_matches_syn(self, r16_r64_syntx):
        """disp_tensor_to_itk(model.warp_l2r) matches fwdtransforms[0]."""
        fi, mi, reg = r16_r64_syntx
        model = reg['model']
        if not hasattr(model, 'warp_l2r'):
            pytest.skip("Model has no warp_l2r")

        # Convert model tensor to ITK via spatial module
        warp_itk = disp_tensor_to_itk(model.warp_l2r, fi)

        # Read the warp file that registration() wrote
        warp_file_itk = ants.image_read(reg['fwdtransforms'][0])

        # Compare: should have same component ordering
        corr = np.corrcoef(
            warp_itk.numpy().flatten(),
            warp_file_itk.numpy().flatten()
        )[0, 1]
        assert corr > 0.99, f"Component order mismatch: correlation={corr:.4f}"

    def test_file_round_trip(self):
        """Write to NIfTI file and read back."""
        fi = ants.image_read(ants.get_data('r16'))
        disp = torch.randn(1, 256, 256, 2) * 2.0
        itk_img = disp_tensor_to_itk(disp, fi)

        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
            path = f.name
        try:
            ants.image_write(itk_img, path)
            recovered = disp_itk_to_tensor(path)
            np.testing.assert_array_almost_equal(
                recovered.numpy(), disp.numpy(), decimal=4
            )
        finally:
            os.unlink(path)

    def test_disp_tensor_to_itk_batched(self):
        """Batched displacement tensor (B > 1) returns a list of ANTsImage objects."""
        fi = ants.image_read(ants.get_data('r16'))
        disp_2d = torch.randn(2, 256, 256, 2)
        itk_imgs = disp_tensor_to_itk(disp_2d, fi)
        assert isinstance(itk_imgs, list)
        assert len(itk_imgs) == 2
        assert all(isinstance(img, ants.ANTsImage) for img in itk_imgs)
        assert itk_imgs[0].components == 2

        fi_3d = ants.from_numpy(np.zeros((16, 16, 16), dtype=np.float32), spacing=(1.0, 1.0, 1.0))
        disp_3d = torch.randn(3, 16, 16, 16, 3)
        itk_imgs_3d = disp_tensor_to_itk(disp_3d, fi_3d)
        assert isinstance(itk_imgs_3d, list)
        assert len(itk_imgs_3d) == 3
        assert all(isinstance(img, ants.ANTsImage) for img in itk_imgs_3d)
        assert itk_imgs_3d[0].components == 3

    def test_disp_itk_to_tensor_sequence(self):
        """Sequence (list/tuple) of ANTsImage objects or paths returns batched tensor (B, *spatial, dim)."""
        fi = ants.image_read(ants.get_data('r16'))
        disp1 = torch.randn(1, 256, 256, 2)
        disp2 = torch.randn(1, 256, 256, 2)
        img1 = disp_tensor_to_itk(disp1, fi)
        img2 = disp_tensor_to_itk(disp2, fi)

        # Test list of ANTsImages
        t_seq = disp_itk_to_tensor([img1, img2])
        assert isinstance(t_seq, torch.Tensor)
        assert t_seq.shape == (2, 256, 256, 2)
        np.testing.assert_array_almost_equal(t_seq[0].numpy(), disp1[0].numpy(), decimal=5)
        np.testing.assert_array_almost_equal(t_seq[1].numpy(), disp2[0].numpy(), decimal=5)

        # Test tuple of file paths
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f2:
            path1, path2 = f1.name, f2.name
        try:
            ants.image_write(img1, path1)
            ants.image_write(img2, path2)
            t_paths = disp_itk_to_tensor((path1, path2))
            assert isinstance(t_paths, torch.Tensor)
            assert t_paths.shape == (2, 256, 256, 2)
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: image_to_tensor / tensor_to_image Round-Trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageConversions:
    def test_round_trip_2d(self):
        img = ants.image_read(ants.get_data('r16'))
        t = image_to_tensor(img)
        assert t.shape == (1, 1, *img.shape)
        recovered = tensor_to_image(t, img)
        np.testing.assert_array_almost_equal(
            recovered.numpy(), img.numpy(), decimal=5
        )

    def test_round_trip_3d(self):
        arr = np.random.randn(32, 32, 32).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.5, 1.5, 1.5))
        t = image_to_tensor(img)
        assert t.shape == (1, 1, 32, 32, 32)
        recovered = tensor_to_image(t, img)
        np.testing.assert_array_almost_equal(
            recovered.numpy(), arr, decimal=5
        )

    def test_metadata_preserved(self):
        img = ants.image_read(ants.get_data('r16'))
        t = image_to_tensor(img)
        recovered = tensor_to_image(t, img)
        assert recovered.spacing == img.spacing
        assert recovered.origin == img.origin


# ═══════════════════════════════════════════════════════════════════════════════
# Test: jacobian_determinant — Validated against ANTs Reference
# ═══════════════════════════════════════════════════════════════════════════════

class TestJacobianDeterminant:
    def test_identity_field_2d(self):
        """Zero displacement should give det(J) = 1.0 everywhere."""
        disp = np.zeros((64, 64, 2), dtype=np.float32)
        detJ = jacobian_determinant(disp, spacing=(1.0, 1.0))
        np.testing.assert_array_almost_equal(detJ, np.ones((64, 64)), decimal=5)

    def test_identity_field_3d(self):
        """Zero displacement should give det(J) = 1.0 everywhere."""
        disp = np.zeros((16, 16, 16, 3), dtype=np.float32)
        detJ = jacobian_determinant(disp, spacing=(1.0, 1.0, 1.0))
        np.testing.assert_array_almost_equal(detJ, np.ones((16, 16, 16)), decimal=5)

    def test_uniform_expansion_2d(self):
        """Uniform scaling field: u_i = alpha * coord_i implies det(J) = (1+alpha)^2."""
        alpha = 0.1
        H, W = 64, 64
        # In axis-aligned convention: comp 0 = displacement along axis 0 (rows/Y)
        #                             comp 1 = displacement along axis 1 (cols/X)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        disp = np.stack([alpha * yy, alpha * xx], axis=-1).astype(np.float32)
        detJ = jacobian_determinant(disp, spacing=(1.0, 1.0))
        expected = (1 + alpha) ** 2
        # Interior points should match (edges have boundary effects from np.gradient)
        np.testing.assert_almost_equal(
            np.mean(detJ[5:-5, 5:-5]), expected, decimal=3
        )

    def test_vs_ants_reference_2d(self, r16_r64_registration):
        """2D Jacobian must match ants.create_jacobian_determinant_image with r > 0.999."""
        fi, mi, reg = r16_r64_registration
        warp_file = reg['fwdtransforms'][0]

        # ANTs reference
        jac_ants = ants.create_jacobian_determinant_image(fi, warp_file, do_log=False)

        # syntx.spatial Jacobian from the same ITK displacement field
        warp_np = ants.image_read(warp_file).numpy()
        detJ = jacobian_determinant(warp_np, ref_image=fi)

        corr = np.corrcoef(detJ.flatten(), jac_ants.numpy().flatten())[0, 1]
        assert corr > 0.999, f"2D Jacobian correlation too low: {corr:.6f}"

    def test_auto_reverse_tensor_input_2d(self, r16_r64_registration):
        """Tensor input should auto-reverse components + transpose spatial and still match ANTs."""
        fi, mi, reg = r16_r64_registration
        warp_file = reg['fwdtransforms'][0]

        jac_ants = ants.create_jacobian_determinant_image(fi, warp_file, do_log=False)

        # Create a proper tensor-domain input from the ITK warp:
        # 1. Reverse components: ITK (dx,dy) → tensor (dy,dx)
        # 2. Transpose spatial: ANTs (X,Y) → tensor (Y,X)
        warp_tensor = disp_itk_to_tensor(ants.image_read(warp_file))

        detJ = jacobian_determinant(warp_tensor, ref_image=fi)
        corr = np.corrcoef(detJ.flatten(), jac_ants.numpy().flatten())[0, 1]
        assert corr > 0.999, f"Auto-reverse tensor Jacobian correlation: {corr:.6f}"

    def test_jacobian_determinant_image_2d(self, r16_r64_registration):
        """jacobian_determinant_image returns properly oriented ANTsImage."""
        fi, mi, reg = r16_r64_registration
        warp_file = reg['fwdtransforms'][0]
        warp_np = ants.image_read(warp_file).numpy()
        jac_img = jacobian_determinant_image(warp_np, fi)
        assert isinstance(jac_img, ants.ANTsImage)
        assert jac_img.spacing == fi.spacing
        assert jac_img.shape == fi.shape

    def test_jacobian_5d_tensor_ref_image_none(self):
        """5D batched tensor (1, Z, Y, X, 3) with ref_image=None should not crash."""
        disp_5d = torch.randn(1, 16, 16, 16, 3) * 0.01
        detJ = jacobian_determinant(disp_5d, ref_image=None)
        assert isinstance(detJ, np.ndarray)
        assert detJ.shape == (16, 16, 16)
        assert np.mean(detJ) > 0.0

    def test_jacobian_5d_batched_multi(self):
        """Batched 5D tensor (B > 1, Z, Y, X, 3) returns stacked array (B, Z, Y, X)."""
        disp_5d = torch.randn(2, 16, 16, 16, 3) * 0.01
        detJ = jacobian_determinant(disp_5d, ref_image=None)
        assert isinstance(detJ, np.ndarray)
        assert detJ.shape == (2, 16, 16, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: deformation_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeformationStats:
    def test_identity_field(self):
        disp = np.zeros((32, 32, 2), dtype=np.float32)
        stats = deformation_stats(disp, spacing=(1.0, 1.0))
        assert stats['folding_pct'] == 0.0
        assert abs(stats['mean_j'] - 1.0) < 1e-5
        assert stats['l2_norm'] == 0.0
        assert stats['mean_displacement'] == 0.0

    def test_nonzero_field(self, r16_r64_registration):
        fi, mi, reg = r16_r64_registration
        warp_np = ants.image_read(reg['fwdtransforms'][0]).numpy()
        stats = deformation_stats(warp_np, ref_image=fi)
        assert 'detJ' in stats
        assert 'min_j' in stats
        assert 'folding_pct' in stats
        assert stats['l2_norm'] > 0
        assert stats['mean_displacement'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Integration with syntx.syn model
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyntxIntegration:
    def test_model_warp_jacobian(self, r16_r64_syntx):
        """Jacobian from model.warp_l2r (tensor domain) matches fwdtransforms Jacobian."""
        fi, mi, reg = r16_r64_syntx
        model = reg['model']
        if not hasattr(model, 'warp_l2r'):
            pytest.skip("Model has no warp_l2r")

        # Jacobian from saved ITK warp file
        warp_file = reg['fwdtransforms'][0]
        jac_ref = ants.create_jacobian_determinant_image(fi, warp_file, do_log=False)

        # Jacobian from model tensor via spatial module (auto-reverses components)
        detJ_tensor = jacobian_determinant(model.warp_l2r, ref_image=fi)

        # Jacobian from disp_tensor_to_itk conversion
        warp_itk = disp_tensor_to_itk(model.warp_l2r, fi)
        detJ_itk = jacobian_determinant(warp_itk.numpy(), ref_image=fi)

        # Both should correlate highly with ANTs reference
        corr_tensor = np.corrcoef(detJ_tensor.flatten(), jac_ref.numpy().flatten())[0, 1]
        corr_itk = np.corrcoef(detJ_itk.flatten(), jac_ref.numpy().flatten())[0, 1]

        assert corr_tensor > 0.99, f"Tensor Jacobian correlation: {corr_tensor:.4f}"
        assert corr_itk > 0.99, f"ITK-converted Jacobian correlation: {corr_itk:.4f}"
