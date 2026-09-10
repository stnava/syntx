import pytest
import numpy as np
import torch
import ants

import syntx.spatial as sp


class TestAdversarialAffinePrimitives:
    """Adversarial challenge of grid_to_physical_affine and physical_to_grid_affine."""

    @pytest.fixture
    def anisotropic_images_3d(self):
        shape = (24, 32, 40)
        spacing = (0.7, 1.3, 2.8)
        origin = (-25.0, 50.0, 10.0)
        direction = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        fixed = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        moving = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        return fixed, moving

    @pytest.fixture
    def distinct_images_3d(self):
        fixed = ants.from_numpy(
            np.zeros((20, 25, 30), dtype=np.float32),
            spacing=(0.8, 1.2, 2.5),
            origin=(-50.0, 100.0, 25.0),
            direction=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        )
        moving = ants.from_numpy(
            np.zeros((35, 28, 22), dtype=np.float32),
            spacing=(1.0, 1.5, 2.0),
            origin=(10.0, -20.0, 40.0),
            direction=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        )
        return fixed, moving

    @pytest.fixture
    def anisotropic_images_2d(self):
        shape = (32, 48)
        spacing = (0.6, 1.5)
        origin = (12.0, -34.0)
        direction = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        fixed = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        moving = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        return fixed, moving

    def test_pure_translation_roundtrip_3d(self, anisotropic_images_3d):
        """Test roundtrip T_grid -> (M_phys, t_phys) -> T_grid' with pure anisotropic translation."""
        fixed, moving = anisotropic_images_3d
        T_grid = np.eye(4, dtype=np.float32)
        T_grid[:3, 3] = [0.2, -0.4, 0.6]  # Distinct translations along x, y, z

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
        T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

        # Fails if translation coordinates are permuted/transposed
        max_err = np.max(np.abs(T_grid - T_grid_rec))
        assert max_err < 1e-5, f"Pure translation roundtrip failed: max_err={max_err}, rec=\n{T_grid_rec}"

    def test_pure_scaling_roundtrip_3d(self, anisotropic_images_3d):
        """Test roundtrip T_grid -> (M_phys, t_phys) -> T_grid' with pure anisotropic scaling."""
        fixed, moving = anisotropic_images_3d
        T_grid = np.diag([1.15, 0.85, 1.05, 1.0]).astype(np.float32)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
        T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

        max_err = np.max(np.abs(T_grid - T_grid_rec))
        assert max_err < 1e-5, f"Pure scaling roundtrip failed: max_err={max_err}, rec=\n{T_grid_rec}"

    def test_full_affine_roundtrip_distinct_geometries(self, distinct_images_3d):
        """Test roundtrip with arbitrary affine matrix and distinct fixed/moving geometries."""
        fixed, moving = distinct_images_3d
        T_grid = np.array([
            [1.05, -0.05, 0.02, 0.3],
            [0.03,  0.98, -0.01, -0.2],
            [-0.02, 0.04, 1.02, 0.4],
            [0.0,   0.0,  0.0,  1.0]
        ], dtype=np.float32)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
        T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

        max_err = np.max(np.abs(T_grid - T_grid_rec))
        assert max_err < 1e-5, f"Distinct geometries affine roundtrip failed: max_err={max_err}, rec=\n{T_grid_rec}"

    def test_full_affine_roundtrip_2d(self, anisotropic_images_2d):
        """Test 2D affine roundtrip with anisotropic spacing and rotation."""
        fixed, moving = anisotropic_images_2d
        theta = 0.25
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        T_grid = np.array([
            [cos_t * 1.1, -sin_t * 0.9, 0.35],
            [sin_t * 1.1,  cos_t * 0.9, -0.45],
            [0.0,          0.0,          1.0]
        ], dtype=np.float32)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
        T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

        max_err = np.max(np.abs(T_grid - T_grid_rec))
        assert max_err < 1e-5, f"2D affine roundtrip failed: max_err={max_err}, rec=\n{T_grid_rec}"

    def test_physical_to_grid_roundtrip_preservation(self, anisotropic_images_3d):
        """Test reverse roundtrip (M_phys, t_phys) -> T_grid -> (M_phys', t_phys')."""
        fixed, moving = anisotropic_images_3d
        M_phys_orig = np.array([
            [1.02, -0.08, 0.03],
            [0.05,  0.95, -0.04],
            [-0.01, 0.06, 1.01]
        ], dtype=np.float32)
        t_phys_orig = np.array([12.5, -8.3, 15.1], dtype=np.float32)

        T_grid = sp.physical_to_grid_affine(M_phys_orig, t_phys_orig, fixed, moving)
        M_phys_rec, t_phys_rec = sp.grid_to_physical_affine(T_grid, fixed, moving)

        err_M = np.max(np.abs(M_phys_orig - M_phys_rec))
        err_t = np.max(np.abs(t_phys_orig - t_phys_rec))
        assert err_M < 1e-5, f"Reverse M_phys roundtrip failed: err_M={err_M}"
        assert err_t < 1e-5, f"Reverse t_phys roundtrip failed: err_t={err_t}"


class TestAdversarialCoordinateGridPrimitives:
    """Adversarial challenge of get_physical_grid_torch and physical_to_normalized_torch."""

    @pytest.mark.parametrize("shape,spacing,origin", [
        ((16, 20, 24), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ((15, 25, 35), (0.8, 1.2, 2.5), (-50.0, 25.0, 10.0)),
        ((22, 33), (0.5, 2.0), (10.0, -20.0)),
    ])
    def test_boundary_mapping_to_unit_cube(self, shape, spacing, origin):
        """Boundary voxels must strictly map to [-1.0, 1.0]."""
        dim = len(shape)
        direction = np.eye(dim, dtype=np.float32)
        grid_phys = sp.get_physical_grid_torch(shape, spacing, origin, direction)
        norm_grid = sp.physical_to_normalized_torch(grid_phys, shape, spacing, origin, direction)

        # Min corner (0, ..., 0)
        c_min = norm_grid[(0,) * (dim + 1)]
        expected_min = torch.full((dim,), -1.0)
        assert torch.allclose(c_min, expected_min, atol=1e-5), f"Min corner mismatch: {c_min} vs {expected_min}"

        # Max corner (-1, ..., -1)
        c_max = norm_grid[(0,) + (-1,) * dim]
        expected_max = torch.full((dim,), 1.0)
        assert torch.allclose(c_max, expected_max, atol=1e-5), f"Max corner mismatch: {c_max} vs {expected_max}"

        # Global range must not exceed [-1.0, 1.0] within numerical tolerance
        assert norm_grid.min().item() >= -1.0 - 1e-5
        assert norm_grid.max().item() <= 1.0 + 1e-5

    def test_boundary_mapping_arbitrary_rotations(self):
        """Boundary mapping under non-trivial orthogonal direction cosines."""
        shape = (14, 18, 22)
        spacing = (0.9, 1.4, 2.1)
        origin = (-30.0, 45.0, -15.0)
        # 90-degree permutation rotation matrix
        rot = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ], dtype=np.float32)

        grid_phys = sp.get_physical_grid_torch(shape, spacing, origin, rot)
        norm_grid = sp.physical_to_normalized_torch(grid_phys, shape, spacing, origin, rot)

        c_min = norm_grid[0, 0, 0, 0]
        c_max = norm_grid[0, -1, -1, -1]
        assert torch.allclose(c_min, torch.tensor([-1.0, -1.0, -1.0]), atol=1e-5)
        assert torch.allclose(c_max, torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)

    def test_cached_normalization_parity(self):
        """Verify parity between physical_to_normalized_torch and physical_to_normalized_torch_cached."""
        shape = (10, 15, 20)
        spacing = (1.2, 0.8, 2.0)
        origin = (5.0, -10.0, 15.0)
        direction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)

        coords = torch.randn(1, 10, 15, 20, 3)
        norm_std = sp.physical_to_normalized_torch(coords, shape, spacing, origin, direction)

        shape_t = torch.tensor(shape, dtype=torch.float32)
        spacing_t = torch.tensor(tuple(reversed(spacing)), dtype=torch.float32)
        origin_t = torch.tensor(tuple(reversed(origin)), dtype=torch.float32)
        direction_t = torch.tensor(direction[::-1, ::-1].copy(), dtype=torch.float32)

        norm_cached = sp.physical_to_normalized_torch_cached(coords, shape_t, spacing_t, origin_t, direction_t)
        max_diff = torch.max(torch.abs(norm_std - norm_cached)).item()
        assert max_diff < 1e-5, f"Cached normalization differs by {max_diff}"


class TestAdversarialAutogradPhysicalScale:
    """Adversarial challenge of compute_autograd_physical_scale with anisotropic acquisitions."""

    def test_anisotropic_scaling_cross_axis_invariance(self):
        """Cross-axis scaling check: X channel must be scaled by X-spacing/shape, not Z."""
        shape_t = torch.tensor([16.0, 32.0, 64.0])    # Z, Y, X
        spacing_t = torch.tensor([3.0, 2.0, 0.5])   # Z, Y, X

        scale = sp.compute_autograd_physical_scale(shape_t, spacing_t)

        # scale_z = (16 - 1) * 3.0 / 2 = 22.5
        # scale_y = (32 - 1) * 2.0 / 2 = 31.0
        # scale_x = (64 - 1) * 0.5 / 2 = 15.75
        # Channel 0 (x) must be 15.75
        # Channel 1 (y) must be 31.0
        # Channel 2 (z) must be 22.5
        expected = torch.tensor([15.75, 31.0, 22.5])
        assert torch.allclose(scale, expected, atol=1e-5)

    def test_2d_anisotropic_scaling(self):
        """2D anisotropic scaling: channel 0 (x) and channel 1 (y)."""
        shape_t = torch.tensor([25.0, 50.0])   # Y, X
        spacing_t = torch.tensor([2.0, 0.4])  # Y, X

        scale = sp.compute_autograd_physical_scale(shape_t, spacing_t)

        # scale_y = (25 - 1) * 2.0 / 2 = 24.0
        # scale_x = (50 - 1) * 0.4 / 2 = 9.8
        # Output channel 0 (x) = 9.8, channel 1 (y) = 24.0
        expected = torch.tensor([9.8, 24.0])
        assert torch.allclose(scale, expected, atol=1e-5)

    def test_python_tuples_input_compatibility(self):
        """Verify compute_autograd_physical_scale accepts Python tuples and lists."""
        scale = sp.compute_autograd_physical_scale((10, 20, 30), [2.5, 2.0, 1.0])
        expected = torch.tensor([14.5, 19.0, 11.25])
        assert torch.allclose(scale, expected, atol=1e-5)
