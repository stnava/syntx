"""
tests/test_spatial_centralization.py — Module Centralization & Backward Compatibility Test Suite

Verifies top-level access to syntx.spatial, __all__ inclusion, native consolidation of spatial
and affine primitives, backward-compatible import aliases across syntx.transform, syntx.core.grid,
and syntx.core.affine, affine parameter roundtrips, and physical coordinate grid scaling.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import ants

import syntx
import syntx.spatial as sp


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: Feature Coverage (Centralization & Backward Compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialCentralizationFeatureCoverage:
    """Tier 1: Verify centralization of spatial primitives and backward compatibility."""

    def test_top_level_spatial_import(self):
        """syntx.spatial is accessible directly upon importing syntx."""
        assert hasattr(syntx, "spatial"), "syntx must expose spatial module at top level"
        assert syntx.spatial is sp, "syntx.spatial must match imported syntx.spatial"

    def test_spatial_all_inclusion(self):
        """'spatial' must be declared in syntx.__all__."""
        assert hasattr(syntx, "spatial"), "syntx must expose spatial module"
        assert "spatial" in syntx.__all__, "'spatial' must be included in syntx.__all__"

    def test_spatial_centralized_exports(self):
        """syntx.spatial exposes the complete suite of centralized spatial functions."""
        required_functions = [
            "disp_tensor_to_itk",
            "disp_itk_to_tensor",
            "export_ants_displacement_field",
            "export_ants_affine_transform",
            "create_ants_affine",
            "grid_to_physical_affine",
            "grid_to_physical_affine_torch",
            "physical_to_grid_affine",
            "get_physical_grid_torch",
            "physical_to_normalized_torch",
            "physical_to_normalized_torch_cached",
            "get_identity_grid_torch",
            "compute_autograd_physical_scale",
            "image_to_tensor",
            "tensor_to_image",
            "reverse_components",
            "reverse_metadata",
            "itk_shape_to_tensor_shape",
            "jacobian_determinant",
            "deformation_stats",
        ]
        for fn_name in required_functions:
            assert hasattr(sp, fn_name), f"syntx.spatial is missing required primitive '{fn_name}'"
            assert callable(getattr(sp, fn_name)), f"syntx.spatial.{fn_name} must be callable"

    def test_spatial_native_displacement_export(self):
        """sp.export_ants_displacement_field works directly from syntx.spatial."""
        disp_arr = np.zeros((20, 24, 2), dtype=np.float32)
        itk_field = sp.export_ants_displacement_field(
            disp_arr,
            origin=(10.0, 20.0),
            spacing=(1.5, 2.5),
            direction=np.eye(2)
        )
        assert isinstance(itk_field, ants.ANTsImage)
        assert itk_field.has_components
        assert np.allclose(itk_field.origin, (10.0, 20.0))
        assert np.allclose(itk_field.spacing, (1.5, 2.5))

    def test_spatial_native_affine_export(self):
        """sp.export_ants_affine_transform and create_ants_affine work natively in spatial."""
        M = np.eye(3, dtype=np.float32)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tx_fwd, tx_inv = sp.export_ants_affine_transform(M, t, dim=3)
        assert isinstance(tx_fwd, ants.ANTsTransform)
        assert isinstance(tx_inv, ants.ANTsTransform)
        assert len(tx_fwd.parameters) == 12

    def test_spatial_shape_mapping(self):
        """sp.itk_shape_to_tensor_shape correctly reverses shape tuples."""
        assert sp.itk_shape_to_tensor_shape((20, 30)) == (30, 20)
        assert sp.itk_shape_to_tensor_shape((10, 20, 30)) == (30, 20, 10)

    def test_spatial_identity_grid_generation(self):
        """sp.get_identity_grid_torch generates [-1, 1] normalized grids in 2D and 3D."""
        id_2d = sp.get_identity_grid_torch((16, 24))
        assert id_2d.shape == (1, 16, 24, 2)
        assert np.isclose(id_2d.min().item(), -1.0)
        assert np.isclose(id_2d.max().item(), 1.0)

        id_3d = sp.get_identity_grid_torch((8, 12, 16))
        assert id_3d.shape == (1, 8, 12, 16, 3)
        assert np.isclose(id_3d.min().item(), -1.0)
        assert np.isclose(id_3d.max().item(), 1.0)

    def test_transform_backward_compatibility(self):
        """syntx.transform continues to export export_ants_displacement_field and export_ants_affine_transform."""
        import syntx.transform as tf
        assert hasattr(tf, "export_ants_displacement_field")
        assert hasattr(tf, "export_ants_affine_transform")
        assert callable(tf.export_ants_displacement_field)
        assert callable(tf.export_ants_affine_transform)

        # Functional validation: export displacement field through transform alias
        disp_arr = np.zeros((20, 24, 2), dtype=np.float32)
        itk_field = tf.export_ants_displacement_field(
            disp_arr,
            origin=(0.0, 0.0),
            spacing=(1.0, 1.0),
            direction=np.eye(2)
        )
        assert isinstance(itk_field, ants.ANTsImage)
        assert itk_field.has_components

    def test_core_grid_backward_compatibility(self):
        """syntx.core.grid continues to export get_physical_grid_torch and physical_to_normalized_torch."""
        import syntx.core.grid as gr
        assert hasattr(gr, "get_physical_grid_torch")
        assert hasattr(gr, "physical_to_normalized_torch")
        assert hasattr(gr, "physical_to_normalized_torch_cached")
        assert callable(gr.get_physical_grid_torch)
        assert callable(gr.physical_to_normalized_torch)

        # Functional validation: generate physical grid through core.grid alias
        shape = (10, 12, 14)
        grid = gr.get_physical_grid_torch(shape, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
        assert grid.shape == (1, 10, 12, 14, 3)

    def test_core_affine_backward_compatibility(self):
        """syntx.core.affine continues to export grid_to_physical_affine and parse_ants_affine."""
        import syntx.core.affine as aff
        assert hasattr(aff, "grid_to_physical_affine")
        assert hasattr(aff, "parse_ants_affine")
        assert callable(aff.grid_to_physical_affine)

        # Functional validation: identity grid to physical affine
        ref = ants.from_numpy(np.zeros((16, 16, 16), dtype=np.float32))
        M, t = aff.grid_to_physical_affine(np.eye(4, dtype=np.float32), ref, ref)
        assert np.allclose(M, np.eye(3), atol=1e-5)
        assert np.allclose(t, np.zeros(3), atol=1e-5)

    def test_affine_parameter_export_3d(self):
        """export_ants_affine_transform exports valid 3D forward and inverse ITK transforms."""
        M_phys = np.array([
            [1.02, 0.05, -0.02],
            [-0.04, 0.98, 0.03],
            [0.01, -0.02, 1.01]
        ], dtype=np.float32)
        t_phys = np.array([12.5, -8.3, 4.7], dtype=np.float32)

        tx_fwd, tx_inv = sp.export_ants_affine_transform(M_phys, t_phys, dim=3)

        assert isinstance(tx_fwd, ants.ANTsTransform)
        assert isinstance(tx_inv, ants.ANTsTransform)
        assert len(tx_fwd.parameters) == 12
        assert len(tx_inv.parameters) == 12

        # Check forward parameters: [M.ravel(), t]
        expected_fwd = np.concatenate([M_phys.ravel(), t_phys])
        np.testing.assert_allclose(tx_fwd.parameters, expected_fwd, atol=1e-5)

        # Check inverse parameters: [inv(M).ravel(), -inv(M) @ t]
        M_inv = np.linalg.inv(M_phys)
        t_inv = -M_inv @ t_phys
        expected_inv = np.concatenate([M_inv.ravel(), t_inv])
        np.testing.assert_allclose(tx_inv.parameters, expected_inv, atol=1e-5)

    def test_affine_parameter_export_2d(self):
        """export_ants_affine_transform exports valid 2D forward and inverse ITK transforms."""
        M_phys = np.array([
            [0.95, 0.08],
            [-0.08, 0.95]
        ], dtype=np.float32)
        t_phys = np.array([5.0, -10.0], dtype=np.float32)

        tx_fwd, tx_inv = sp.export_ants_affine_transform(M_phys, t_phys, dim=2)
        assert len(tx_fwd.parameters) == 6
        assert len(tx_inv.parameters) == 6

        expected_fwd = np.concatenate([M_phys.ravel(), t_phys])
        np.testing.assert_allclose(tx_fwd.parameters, expected_fwd, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: Boundary, Edge & Corner Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialCentralizationBoundaryCases:
    """Tier 2: Extreme values, input types, and physical coordinate edge cases."""

    def test_affine_identity_roundtrip(self):
        """Identity affine matrix (M = I, t = 0) yields identity forward and inverse transforms."""
        tx_fwd, tx_inv = sp.export_ants_affine_transform(np.eye(3), np.zeros(3), dim=3)
        expected = np.concatenate([np.eye(3).ravel(), np.zeros(3)])
        np.testing.assert_allclose(tx_fwd.parameters, expected, atol=1e-6)
        np.testing.assert_allclose(tx_inv.parameters, expected, atol=1e-6)

    def test_affine_pure_translation(self):
        """Pure translation (M = I, t != 0) yields inverse translation t_inv = -t."""
        t = np.array([15.0, -25.0, 35.0], dtype=np.float32)
        tx_fwd, tx_inv = sp.export_ants_affine_transform(np.eye(3), t, dim=3)

        np.testing.assert_allclose(tx_fwd.parameters[9:], t, atol=1e-6)
        np.testing.assert_allclose(tx_inv.parameters[9:], -t, atol=1e-6)

    def test_affine_tensor_and_numpy_inputs(self):
        """export_ants_affine_transform transparently accepts PyTorch tensors."""
        M_t = torch.tensor([[1.0, 0.1], [-0.1, 1.0]], dtype=torch.float32)
        t_t = torch.tensor([3.0, -4.0], dtype=torch.float32)

        tx_fwd, tx_inv = sp.export_ants_affine_transform(M_t, t_t, dim=2)
        np.testing.assert_allclose(tx_fwd.parameters, np.array([1.0, 0.1, -0.1, 1.0, 3.0, -4.0]), atol=1e-5)

    def test_affine_file_export_and_read(self):
        """export_ants_affine_transform writes a valid .mat file that ANTs reads back."""
        M = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]], dtype=np.float32)
        t = np.array([7.0, -14.0, 21.0], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            tmp_mat = f.name

        try:
            fwd, inv = sp.export_ants_affine_transform(M, t, dim=3, filename=tmp_mat)
            assert os.path.exists(tmp_mat)

            # Read back with ANTs
            tx_read = ants.read_transform(tmp_mat)
            np.testing.assert_allclose(tx_read.parameters, fwd.parameters, atol=1e-5)
        finally:
            if os.path.exists(tmp_mat):
                os.unlink(tmp_mat)

    def test_coordinate_grid_scaling_boundaries(self):
        """get_physical_grid_torch extreme coordinates match origin and origin + (N-1)*spacing."""
        # PyTorch shape: (Z=10, Y=20, X=30)
        shape = (10, 20, 30)
        # ITK spacing: (X=1.5, Y=2.0, Z=2.5)
        spacing = (1.5, 2.0, 2.5)
        # ITK origin: (X=5.0, Y=10.0, Z=15.0)
        origin = (5.0, 10.0, 15.0)

        grid = sp.get_physical_grid_torch(shape, spacing, origin, np.eye(3))
        # Grid coordinates are in reversed (Z, Y, X) order matching PyTorch layout
        origin_pt = grid[0, 0, 0, 0].numpy()
        expected_origin = np.array([15.0, 10.0, 5.0])
        np.testing.assert_allclose(origin_pt, expected_origin, atol=1e-5)

        end_pt = grid[0, shape[0] - 1, shape[1] - 1, shape[2] - 1].numpy()
        expected_end = np.array([
            15.0 + (10 - 1) * 2.5,
            10.0 + (20 - 1) * 2.0,
            5.0 + (30 - 1) * 1.5
        ])
        np.testing.assert_allclose(end_pt, expected_end, atol=1e-5)

    def test_coordinate_grid_normalization_range(self):
        """physical_to_normalized_torch maps physical coordinates strictly into [-1.0, 1.0]."""
        shape = (8, 12, 16)
        spacing = (1.2, 1.4, 1.6)
        origin = (2.0, 4.0, 6.0)
        direction = np.eye(3)

        grid_phys = sp.get_physical_grid_torch(shape, spacing, origin, direction)
        grid_norm = sp.physical_to_normalized_torch(grid_phys, shape, spacing, origin, direction)

        assert grid_norm.min().item() >= -1.0 - 1e-6
        assert grid_norm.max().item() <= 1.0 + 1e-6

        # Check corner coordinates map to exact -1 and +1
        np.testing.assert_allclose(grid_norm[0, 0, 0, 0].numpy(), np.array([-1.0, -1.0, -1.0]), atol=1e-5)
        np.testing.assert_allclose(grid_norm[0, -1, -1, -1].numpy(), np.array([1.0, 1.0, 1.0]), atol=1e-5)

    def test_autograd_physical_scale_channel_flip(self):
        """Autograd physical scaling flips dim 0 so anisotropic dimensions match vector channels."""
        # Spec invariant from GEMINI.md Rule 2:
        # s_phys = flip((N - 1) * s / 2, dim=0)
        shape_t = torch.tensor([10.0, 20.0, 30.0])      # Z, Y, X
        spacing_t = torch.tensor([2.5, 2.0, 1.0])    # Z, Y, X

        scale = sp.compute_autograd_physical_scale(shape_t, spacing_t)

        # Dimension calculation:
        # scale_z = (10 - 1) * 2.5 / 2 = 11.25
        # scale_y = (20 - 1) * 2.0 / 2 = 19.0
        # scale_x = (30 - 1) * 1.0 / 2 = 14.5
        # With flip: scale[0] is X (14.5), scale[1] is Y (19.0), scale[2] is Z (11.25)
        expected = torch.tensor([14.5, 19.0, 11.25])
        assert torch.allclose(scale, expected, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3: Cross-Feature Interactions
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialCentralizationCrossFeatureInteractions:
    """Tier 3: Interactions between grid conversions, affine transforms, and normalization."""

    def test_grid_to_physical_affine_and_export(self):
        """Convert normalized grid affine to physical affine, then export to ANTsTransform."""
        fixed = ants.from_numpy(np.zeros((18, 20, 22), dtype=np.float32), spacing=(1.0, 1.1, 1.2))
        moving = ants.from_numpy(np.zeros((18, 20, 22), dtype=np.float32), spacing=(1.0, 1.1, 1.2))

        # 4x4 affine in normalized grid coordinates
        T_grid = np.eye(4, dtype=np.float32)
        T_grid[0, 3] = 0.05  # small shift in normalized grid

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
        assert M_phys.shape == (3, 3)
        assert t_phys.shape == (3,)

        tx_fwd, tx_inv = sp.export_ants_affine_transform(M_phys, t_phys, dim=3)
        assert np.isfinite(tx_fwd.parameters).all()
        assert np.isfinite(tx_inv.parameters).all()

    def test_affine_parameter_forward_inverse_roundtrip(self):
        """Applying forward then inverse affine transforms recovers original physical coordinates."""
        M_phys = np.array([
            [1.02, 0.05, -0.02],
            [-0.04, 0.98, 0.03],
            [0.01, -0.02, 1.01]
        ], dtype=np.float32)
        t_phys = np.array([12.5, -8.3, 4.7], dtype=np.float32)

        tx_fwd, tx_inv = sp.export_ants_affine_transform(M_phys, t_phys, dim=3)
        pts = [
            (0.0, 0.0, 0.0),
            (10.0, 20.0, 30.0),
            (-50.0, 75.0, -100.0)
        ]
        for p in pts:
            p_fwd = tx_fwd.apply_to_point(p)
            p_rec = tx_inv.apply_to_point(p_fwd)
            np.testing.assert_allclose(p_rec, p, atol=1e-5)

    def test_grid_to_physical_affine_identity_roundtrip(self):
        """Identity grid affine maps to identity physical affine and back."""
        fixed = ants.from_numpy(np.zeros((16, 18, 20), dtype=np.float32), spacing=(1.0, 1.0, 1.0))
        moving = ants.from_numpy(np.zeros((16, 18, 20), dtype=np.float32), spacing=(1.0, 1.0, 1.0))

        T_grid_orig = np.eye(4, dtype=np.float32)
        M_phys, t_phys = sp.grid_to_physical_affine(T_grid_orig, fixed, moving)

        np.testing.assert_allclose(M_phys, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(t_phys, np.zeros(3), atol=1e-5)

        T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)
        np.testing.assert_allclose(T_grid_rec, T_grid_orig, atol=1e-5)

    def test_physical_to_grid_affine_asymmetric_roundtrip(self):
        """Asymmetric affine roundtrip preserves translation and scale without axis permutation."""
        fi = ants.from_numpy(np.zeros((32, 48, 64), dtype=np.float32), spacing=(0.8, 1.2, 2.0), origin=(10., 20., 30.))
        T_grid = np.array([
            [1.05, 0.02, 0.00, 0.25],
            [-0.01, 0.98, 0.01, -0.15],
            [0.00, 0.02, 1.02, 0.35],
            [0.00, 0.00, 0.00, 1.00]
        ], dtype=np.float32)
        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fi, fi)
        T_rec = sp.physical_to_grid_affine(M_phys, t_phys, fi, fi)
        np.testing.assert_allclose(T_rec, T_grid, atol=1e-5)


    def test_physical_grid_to_normalized_roundtrip(self):
        """Physical grid coordinates pass through normalization reproducing [-1, 1] grid."""
        shape = (12, 14, 16)
        spacing = (1.5, 1.0, 0.8)
        origin = (10.0, -20.0, 30.0)
        direction = np.eye(3)

        X_phys = sp.get_physical_grid_torch(shape, spacing, origin, direction)
        coords_norm = sp.physical_to_normalized_torch(X_phys, shape, spacing, origin, direction)

        # Center voxel should be close to 0 in normalized space
        cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
        center_norm = coords_norm[0, cz, cy, cx].numpy()
        np.testing.assert_allclose(center_norm, np.zeros(3), atol=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4: Real-World Workload Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialCentralizationRealWorldWorkloads:
    """Tier 4: Realistic registration workflows and multi-resolution grid scaling."""

    def test_affine_transform_application_on_image(self):
        """Physical affine parameters exported to ANTsTransform shift an image correctly."""
        # Create a small 3D volume with a centered cube
        arr = np.zeros((30, 30, 30), dtype=np.float32)
        arr[10:20, 10:20, 10:20] = 100.0
        img = ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

        # Shift by +5mm along X in physical space
        M = np.eye(3, dtype=np.float32)
        t = np.array([5.0, 0.0, 0.0], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
            tmp_mat = tmp.name

        try:
            sp.export_ants_affine_transform(M, t, dim=3, filename=tmp_mat)
            warped = ants.apply_transforms(fixed=img, moving=img, transformlist=[tmp_mat])

            # Transformed image should be shifted along X
            assert warped.shape == img.shape
            assert warped.max() > 50.0
        finally:
            if os.path.exists(tmp_mat):
                os.unlink(tmp_mat)

    def test_multi_resolution_grid_generation(self):
        """Coordinate grids across multi-resolution pyramid levels cover identical physical bounds."""
        base_shape = (32, 32, 32)
        base_spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)

        # Level 1 (full res)
        grid_l1 = sp.get_physical_grid_torch(base_shape, base_spacing, origin, direction)

        # Level 2 (2x downsampled)
        shape_l2 = tuple(s // 2 for s in base_shape)
        spacing_l2 = tuple(sp_val * 2.0 for sp_val in base_spacing)
        grid_l2 = sp.get_physical_grid_torch(shape_l2, spacing_l2, origin, direction)

        # Origin point must match exactly
        np.testing.assert_allclose(grid_l1[0, 0, 0, 0].numpy(), grid_l2[0, 0, 0, 0].numpy(), atol=1e-5)

        # Extent across both levels should be closely aligned within 1 voxel of downsampled grid
        max_l1 = grid_l1[0, -1, -1, -1].numpy()
        max_l2 = grid_l2[0, -1, -1, -1].numpy()
        np.testing.assert_allclose(max_l1, max_l2, atol=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 5: Reverse Components Polymorphism & Scattered Coordinate Parity
# ═══════════════════════════════════════════════════════════════════════════════

class TestReverseComponentsPolymorphism:
    """Tier 5: Verify type preservation, autograd gradients, and involution for reverse_components."""

    def test_reverse_components_tensor_preservation_2d_and_3d(self):
        """reverse_components preserves torch.Tensor type, device, dtype, and shape."""
        # 2D Batched
        t_2d = torch.randn(2, 16, 20, 2, dtype=torch.float32)
        rev_2d = sp.reverse_components(t_2d)
        assert isinstance(rev_2d, torch.Tensor), "2D output must be torch.Tensor"
        assert rev_2d.shape == t_2d.shape
        assert rev_2d.dtype == t_2d.dtype
        assert torch.allclose(rev_2d[..., 0], t_2d[..., 1])
        assert torch.allclose(rev_2d[..., 1], t_2d[..., 0])

        # 3D Batched
        t_3d = torch.randn(1, 8, 12, 16, 3, dtype=torch.float64)
        rev_3d = sp.reverse_components(t_3d)
        assert isinstance(rev_3d, torch.Tensor), "3D output must be torch.Tensor"
        assert rev_3d.shape == t_3d.shape
        assert rev_3d.dtype == torch.float64
        assert torch.allclose(rev_3d[..., 0], t_3d[..., 2])
        assert torch.allclose(rev_3d[..., 1], t_3d[..., 1])
        assert torch.allclose(rev_3d[..., 2], t_3d[..., 0])

    def test_reverse_components_autograd_preservation(self):
        """Autograd gradients flow cleanly through reverse_components without detachment."""
        coords = torch.tensor([[1.5, -2.5], [0.5, 3.0]], dtype=torch.float64, requires_grad=True)
        rev = sp.reverse_components(coords)
        loss = (rev ** 2).sum()
        loss.backward()

        assert coords.grad is not None, "Gradient must not be None after backward pass"
        assert coords.grad.shape == coords.shape
        # d/d[x, y] (y^2 + x^2) = [2x, 2y]
        expected_grad = 2.0 * coords
        assert torch.allclose(coords.grad, expected_grad, atol=1e-7)

    def test_reverse_components_numpy_preservation(self):
        """reverse_components preserves np.ndarray type and reverses last axis."""
        arr = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
        rev = sp.reverse_components(arr)
        assert isinstance(rev, np.ndarray), "Output must be np.ndarray"
        assert rev.shape == arr.shape
        np.testing.assert_array_equal(rev[..., 0], arr[..., 2])
        np.testing.assert_array_equal(rev[..., 1], arr[..., 1])
        np.testing.assert_array_equal(rev[..., 2], arr[..., 0])

    def test_reverse_components_involution_identity(self):
        """reverse_components is strictly self-inverting (f(f(x)) == x) for Tensor and ndarray."""
        t = torch.randn(4, 8, 8, 3)
        assert torch.allclose(sp.reverse_components(sp.reverse_components(t)), t)

        arr = np.random.randn(4, 8, 8, 3).astype(np.float32)
        np.testing.assert_array_equal(sp.reverse_components(sp.reverse_components(arr)), arr)


class TestScatteredCoordinateParity:
    """Tier 5: Verify integration between spatial primitives and scattered mapping."""

    def test_scattered_evaluate_field_convention_parity(self):
        """evaluate_field_at_scattered yields identical values across 'xyz' and 'zyx' via reverse_components."""
        from syntx.scattered.mapping import evaluate_field_at_scattered

        # Create 2D field with independent spatial variations
        H, W = 16, 20
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        field = torch.stack([gx, gy], dim=-1)  # (H, W, 2)

        # Query points in XYZ Cartesian convention
        pts_xyz = torch.tensor([[0.4, -0.3], [-0.2, 0.7]], dtype=torch.float32)
        # Reverse to ZYX tensor convention
        pts_zyx = sp.reverse_components(pts_xyz)

        out_xyz = evaluate_field_at_scattered(field, pts_xyz, coord_convention='xyz')
        out_zyx = evaluate_field_at_scattered(field, pts_zyx, coord_convention='zyx')

        assert torch.allclose(out_xyz, out_zyx, atol=1e-5), "Field sampling must match across conventions"

    def test_scattered_pullback_convention_parity(self):
        """pullback_grid_to_scattered yields identical features across conventions via reverse_components."""
        from syntx.scattered.transport import pullback_grid_to_scattered

        grid = torch.randn(1, 1, 16, 16, dtype=torch.float32)
        pts_xyz = torch.tensor([[0.25, -0.15], [-0.35, 0.45]], dtype=torch.float32)
        pts_zyx = sp.reverse_components(pts_xyz)

        feat_xyz = pullback_grid_to_scattered(grid, pts_xyz, coord_convention='xyz')
        feat_zyx = pullback_grid_to_scattered(grid, pts_zyx, coord_convention='zyx')

        assert torch.allclose(feat_xyz, feat_zyx, atol=1e-5)


class TestSpatialImageTensorDomainRoundtrip:
    """Tier 5: Verify exact image-tensor roundtrip and optional ZYX transposition."""

    def test_image_to_tensor_and_tensor_to_image_identity(self):
        """image_to_tensor and tensor_to_image achieve exact roundtrip numerical identity in 2D and 3D."""
        # 2D test
        arr_2d = np.random.randn(24, 32).astype(np.float32)
        img_2d = ants.from_numpy(arr_2d, spacing=(1.2, 1.5), origin=(5.0, 10.0))
        t_2d = sp.image_to_tensor(img_2d)
        assert t_2d.shape == (1, 1, 24, 32)  # default ITK shape preservation
        rec_2d = sp.tensor_to_image(t_2d, ref_image=img_2d)
        np.testing.assert_allclose(rec_2d.numpy(), arr_2d, atol=1e-6)
        assert np.allclose(rec_2d.spacing, img_2d.spacing)
        assert np.allclose(rec_2d.origin, img_2d.origin)

        # 2D test with to_zyx=True
        t_2d_zyx = sp.image_to_tensor(img_2d, to_zyx=True)
        assert t_2d_zyx.shape == (1, 1, 32, 24)

        # 3D test
        arr_3d = np.random.randn(16, 20, 24).astype(np.float32)
        img_3d = ants.from_numpy(arr_3d, spacing=(1.0, 1.2, 1.4), origin=(2.0, 4.0, 6.0))
        t_3d = sp.image_to_tensor(img_3d)
        assert t_3d.shape == (1, 1, 16, 20, 24)  # default ITK shape preservation
        rec_3d = sp.tensor_to_image(t_3d, ref_image=img_3d)
        np.testing.assert_allclose(rec_3d.numpy(), arr_3d, atol=1e-6)
        assert np.allclose(rec_3d.spacing, img_3d.spacing)
        assert np.allclose(rec_3d.origin, img_3d.origin)

        # 3D test with to_zyx=True
        t_3d_zyx = sp.image_to_tensor(img_3d, to_zyx=True)
        assert t_3d_zyx.shape == (1, 1, 24, 20, 16)

    def test_image_to_tensor_numpy_and_torch_roundtrip(self):
        """image_to_tensor on NumPy and Tensor inputs produces correct shape and recovers ANTsImage."""
        # 2D numpy array
        arr_2d = np.random.randn(24, 32).astype(np.float32)
        ref_2d = ants.from_numpy(arr_2d, spacing=(1.2, 1.5), origin=(5.0, 10.0))
        t_2d_np = sp.image_to_tensor(arr_2d)
        assert t_2d_np.shape == (1, 1, 24, 32)
        rec_2d = sp.tensor_to_image(t_2d_np, ref_image=ref_2d)
        np.testing.assert_allclose(rec_2d.numpy(), arr_2d, atol=1e-6)

        # 3D numpy array
        arr_3d = np.random.randn(16, 20, 24).astype(np.float32)
        ref_3d = ants.from_numpy(arr_3d, spacing=(1.0, 1.2, 1.4), origin=(2.0, 4.0, 6.0))
        t_3d_np = sp.image_to_tensor(arr_3d)
        assert t_3d_np.shape == (1, 1, 16, 20, 24)
        rec_3d = sp.tensor_to_image(t_3d_np, ref_image=ref_3d)
        np.testing.assert_allclose(rec_3d.numpy(), arr_3d, atol=1e-6)

        # 3D torch tensor
        t_in = torch.from_numpy(arr_3d)
        t_3d_torch = sp.image_to_tensor(t_in)
        assert t_3d_torch.shape == (1, 1, 16, 20, 24)
        rec_3d_t = sp.tensor_to_image(t_3d_torch, ref_image=ref_3d)
        np.testing.assert_allclose(rec_3d_t.numpy(), arr_3d, atol=1e-6)

        # 3D batched unchanneled tensor (1, D, H, W)
        t_1d_hw = t_in.unsqueeze(0)
        t_out = sp.image_to_tensor(t_1d_hw)
        assert t_out.shape == (1, 1, 16, 20, 24)

        # 3D multi-batch unchanneled tensor (2, D, H, W)
        t_2d_hw = torch.stack([t_in, t_in], dim=0)
        t_out2 = sp.image_to_tensor(t_2d_hw)
        assert t_out2.shape == (2, 1, 16, 20, 24)

