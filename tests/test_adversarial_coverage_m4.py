"""
tests/test_adversarial_coverage_m4.py

Feature 23: Adversarial Coverage Hardening Suite (Milestone 4).
Exhaustively stress-tests white-box edge cases for physical space centralization:
  1. Extreme spatial anisotropies (voxel spacing ratio > 10:1, e.g. (0.2, 0.2, 3.0)).
  2. Non-trivial oblique direction matrices with determinant -1 (reflections) and +1 (rotations).
  3. Batched displacement field inversion and affine composition roundtrips (B >= 4).
  4. Numerical roundtrip invariance under float32 and float64 precisions.
"""

import pytest
import numpy as np
import torch
import ants

import syntx.spatial as sp
from syntx.core.grid import compose_grids, get_identity_grid_torch
from syntx.core.inverse import update_inverse_field_nd_anderson, compute_inverse_identity_error_nd


def _make_3d_rotation_matrix(angle_x, angle_y, angle_z):
    """Constructs a 3D rotation matrix in SO(3) with det = +1.0."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    R = Rz @ Ry @ Rx
    return R


class TestAdversarialExtremeAnisotropy:
    """Stress tests on extreme spatial anisotropies with voxel spacing ratios > 10:1."""

    def test_extreme_anisotropy_ratio_15_to_1_grid_mapping(self):
        """Tests 3D physical grid generation and normalization with 15:1 anisotropy (0.2, 0.2, 3.0 mm)."""
        shape = (16, 24, 32)
        spacing = (0.2, 0.2, 3.0)  # Ratio 3.0 / 0.2 = 15.0 > 10.0
        origin = (-20.0, 15.0, 45.0)
        direction = np.eye(3)

        # 1. Physical coordinate grid generation
        phys_grid = sp.get_physical_grid_torch(shape, spacing, origin, direction)
        assert phys_grid.shape == (1, 16, 24, 32, 3)

        # 2. Map physical coordinates back to normalized [-1, 1] grid
        norm_grid = sp.physical_to_normalized_torch(phys_grid, shape, spacing, origin, direction)
        assert norm_grid.min().item() >= -1.0 - 1e-5
        assert norm_grid.max().item() <= 1.0 + 1e-5

        # Canonical identity grid comparison
        id_grid = sp.get_identity_grid_torch(shape)
        max_err = (norm_grid - id_grid).abs().max().item()
        assert max_err < 1e-5, f"Normalized grid deviation on 15:1 anisotropy: {max_err}"

    def test_extreme_anisotropy_autograd_physical_scale_channel_flip(self):
        """
        Tests GEMINI.md Rule 2 invariant under extreme 25:1 anisotropy (0.1, 0.2, 2.5 mm).
        Verifies that PyTorch autograd physical scale tensor flips dimension 0 (torch.flip),
        guaranteeing x-displacements scale with x-dimension/spacing rather than z-dimension.
        """
        shape_t = torch.tensor([12, 20, 28], dtype=torch.float32)  # Z, Y, X
        spacing_t = torch.tensor([2.5, 0.2, 0.1], dtype=torch.float32)  # z_sp, y_sp, x_sp

        scale = sp.compute_autograd_physical_scale(shape_t, spacing_t)
        assert scale.shape == (3,)

        # Expected: flipped ordering (N - 1) * s / 2
        # Component 0 (x) must scale by (N_x - 1) * s_x / 2 = (28 - 1) * 0.1 / 2 = 1.35
        # Component 2 (z) must scale by (N_z - 1) * s_z / 2 = (12 - 1) * 2.5 / 2 = 13.75
        expected_x_scale = (28 - 1) * 0.1 / 2.0
        expected_y_scale = (20 - 1) * 0.2 / 2.0
        expected_z_scale = (12 - 1) * 2.5 / 2.0

        assert torch.isclose(scale[0], torch.tensor(expected_x_scale), atol=1e-5)
        assert torch.isclose(scale[1], torch.tensor(expected_y_scale), atol=1e-5)
        assert torch.isclose(scale[2], torch.tensor(expected_z_scale), atol=1e-5)

        # Cross-axis failure check: non-flipped scale would give ~13.75 for component 0 (10x error!)
        wrong_x_scale = (12 - 1) * 2.5 / 2.0
        assert abs(scale[0].item() - wrong_x_scale) > 10.0, "Channel flip check failed: scaling was inverted!"

    def test_extreme_anisotropy_affine_roundtrip(self):
        """Verifies T_grid <-> (M_phys, t_phys) roundtrip with 25:1 spacing ratio."""
        shape = (14, 18, 26)
        spacing = (0.1, 0.2, 2.5)
        origin = (5.0, -10.0, 20.0)
        direction = np.eye(3, dtype=np.float32)

        fixed = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        moving = fixed.clone()

        # Non-trivial grid affine (scaling + rotation + translation)
        T_grid = np.array([
            [1.08, -0.04,  0.02,  0.25],
            [0.03,  0.95, -0.03, -0.30],
            [-0.01, 0.02,  1.05,  0.15],
            [0.0,   0.0,   0.0,   1.0]
        ], dtype=np.float32)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
        T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

        max_err = np.max(np.abs(T_grid - T_grid_rec))
        assert max_err < 1e-5, f"Affine roundtrip error on extreme anisotropy: {max_err}"

    def test_extreme_anisotropy_2d_ratio_20_to_1(self):
        """Verifies 2D displacement field roundtrip with 20:1 anisotropy (0.15, 3.0 mm)."""
        shape = (16, 64)
        spacing = (0.15, 3.0)
        origin = (10.0, -25.0)
        direction = np.eye(2)

        ref_img = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=direction)
        disp_tensor = torch.randn(1, 64, 16, 2, dtype=torch.float32)

        itk_field = sp.disp_tensor_to_itk(disp_tensor, ref_img)
        rec_tensor = sp.disp_itk_to_tensor(itk_field)

        assert rec_tensor.shape == disp_tensor.shape
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err == 0.0, f"Expected exact bitwise identity on 2D 20:1 anisotropy, got Linf={linf_err}"


class TestAdversarialObliqueDirections:
    """Stress tests on non-trivial oblique direction matrices with determinant +1 and -1."""

    def test_oblique_direction_det_positive_one_rotation(self):
        """Tests continuous 3D rotation in SO(3) with det = +1.0."""
        R = _make_3d_rotation_matrix(0.35, -0.42, 0.61)
        det_R = np.linalg.det(R)
        assert np.isclose(det_R, 1.0), f"Expected det(R) = +1.0, got {det_R}"

        shape = (12, 16, 20)
        spacing = (0.9, 1.1, 1.3)
        origin = (-30.0, 15.0, 50.0)

        ref_img = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=R)

        disp_tensor = torch.randn(1, 20, 16, 12, 3, dtype=torch.float32)
        itk_field = sp.disp_tensor_to_itk(disp_tensor, ref_img)
        rec_tensor = sp.disp_itk_to_tensor(itk_field)

        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err == 0.0, f"Displacement roundtrip failed with oblique SO(3) direction: Linf={linf_err}"

        # Affine roundtrip with oblique SO(3) direction
        T_grid = np.array([
            [1.02, 0.03, -0.01, 0.1],
            [-0.02, 0.98, 0.02, -0.2],
            [0.01, -0.02, 1.03, 0.3],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, ref_img, ref_img)
        T_rec = sp.physical_to_grid_affine(M_phys, t_phys, ref_img, ref_img)
        max_err = np.max(np.abs(T_grid - T_rec))
        assert max_err < 1e-5, f"Affine roundtrip failed with oblique SO(3) direction: {max_err}"

    def test_oblique_direction_det_negative_one_reflection(self):
        """
        Tests non-trivial oblique direction matrix with reflection (det = -1.0).
        Represents radiological/neurological scanner orientations with flipped axes.
        """
        # Oblique reflection: rotation R multiplied by reflection diag(-1, 1, 1)
        R = _make_3d_rotation_matrix(0.25, 0.50, -0.30)
        D_refl = R @ np.diag([-1.0, 1.0, 1.0])
        det_refl = np.linalg.det(D_refl)
        assert np.isclose(det_refl, -1.0), f"Expected det(D) = -1.0, got {det_refl}"

        shape = (14, 18, 22)
        spacing = (0.8, 1.0, 1.2)
        origin = (100.0, -50.0, 25.0)

        ref_img = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin, direction=D_refl)

        # 1. Displacement roundtrip
        disp_tensor = torch.randn(1, 22, 18, 14, 3, dtype=torch.float32)
        itk_field = sp.disp_tensor_to_itk(disp_tensor, ref_img)
        rec_tensor = sp.disp_itk_to_tensor(itk_field)

        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err == 0.0, f"Displacement roundtrip failed with det = -1 reflection: Linf={linf_err}"

        # 2. Coordinate grid mapping
        phys_grid = sp.get_physical_grid_torch(shape, spacing, origin, D_refl)
        norm_grid = sp.physical_to_normalized_torch(phys_grid, shape, spacing, origin, D_refl)
        id_grid = sp.get_identity_grid_torch(shape)
        grid_err = (norm_grid - id_grid).abs().max().item()
        assert grid_err < 1e-5, f"Grid normalization failed with det = -1 reflection: {grid_err}"

        # 3. Affine roundtrip with reflection direction
        T_grid = np.array([
            [1.05, -0.02, 0.01, 0.15],
            [0.01, 0.96, -0.03, -0.10],
            [-0.03, 0.01, 1.02, 0.20],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid, ref_img, ref_img)
        T_rec = sp.physical_to_grid_affine(M_phys, t_phys, ref_img, ref_img)
        aff_err = np.max(np.abs(T_grid - T_rec))
        assert aff_err < 1e-5, f"Affine roundtrip failed with det = -1 reflection: {aff_err}"

    def test_2d_oblique_rotation_and_reflection(self):
        """Tests 2D oblique rotation (det = +1) and 2D oblique reflection (det = -1)."""
        theta = 0.42 * np.pi
        c, s = np.cos(theta), np.sin(theta)

        # 2D Rotation: det = +1
        D_rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        assert np.isclose(np.linalg.det(D_rot), 1.0)

        # 2D Reflection: det = -1
        D_refl = np.array([[c, s], [s, -c]], dtype=np.float64)
        assert np.isclose(np.linalg.det(D_refl), -1.0)

        for D in [D_rot, D_refl]:
            ref = ants.from_numpy(np.zeros((20, 30), dtype=np.float32), spacing=(0.7, 1.4), direction=D)
            disp = torch.randn(1, 30, 20, 2, dtype=torch.float32)

            itk = sp.disp_tensor_to_itk(disp, ref)
            rec = sp.disp_itk_to_tensor(itk)
            assert (rec - disp).abs().max().item() == 0.0


class TestAdversarialBatchedTransforms:
    """Stress tests on batched displacement fields and affine composition (B >= 4)."""

    @pytest.mark.parametrize("batch_size", [4, 6, 8])
    def test_batched_disp_tensor_roundtrip(self, batch_size):
        """Verifies disp_tensor_to_itk <-> disp_itk_to_tensor roundtrip for B in [4, 6, 8]."""
        ref_image = ants.from_numpy(
            np.zeros((16, 20, 24), dtype=np.float32),
            spacing=(1.0, 1.2, 1.4),
            origin=(10.0, -15.0, 20.0),
            direction=_make_3d_rotation_matrix(0.2, -0.1, 0.3)
        )

        disp_tensor = torch.randn(batch_size, 24, 20, 16, 3, dtype=torch.float32)

        # disp_tensor_to_itk returns a list of ANTsImages when B > 1
        itk_list = sp.disp_tensor_to_itk(disp_tensor, ref_image)
        assert isinstance(itk_list, list)
        assert len(itk_list) == batch_size

        for img in itk_list:
            assert isinstance(img, ants.ANTsImage)
            assert img.shape == ref_image.shape

        # disp_itk_to_tensor reconstructs the (B, *spatial, dim) tensor
        rec_tensor = sp.disp_itk_to_tensor(itk_list)
        assert rec_tensor.shape == disp_tensor.shape

        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err == 0.0, f"Batched displacement roundtrip failed (B={batch_size}): Linf={linf_err}"

    def test_batched_displacement_field_inversion_b4(self):
        """Verifies fixed-point Anderson inversion on a batched displacement tensor (B = 4)."""
        torch.manual_seed(42)
        # Small displacement fields for stable inversion
        B = 4
        shape = (16, 16)
        disp = torch.randn(B, *shape, 2, dtype=torch.float32) * 0.03

        # Compute inverse via Anderson acceleration
        inv_disp = update_inverse_field_nd_anderson(disp, -disp, steps=15, m=3)
        assert inv_disp.shape == (B, *shape, 2)

        # Compute composed inverse identity error: ||phi(x + psi(x)) + psi(x)||
        inv_err = compute_inverse_identity_error_nd(disp, inv_disp)
        assert inv_err.shape == (B, *shape)

        mean_err_per_batch = inv_err.mean(dim=[-2, -1])
        for b in range(B):
            assert mean_err_per_batch[b].item() < 0.05, (
                f"Batch item {b} inverse error too high: {mean_err_per_batch[b].item()}"
            )

    def test_batched_affine_composition_roundtrip_b4(self):
        """Verifies composition and inversion of batched 3D affine transformation matrices (B = 4)."""
        B = 4
        rng = np.random.RandomState(42)

        for b in range(B):
            # Generate random invertible affine matrix
            A = rng.randn(3, 3) * 0.1 + np.eye(3)
            t = rng.randn(3) * 5.0
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = A
            T[:3, 3] = t

            T_inv = np.linalg.inv(T)

            # Compositions: T @ T_inv == I and T_inv @ T == I
            composed_fwd = T @ T_inv
            composed_inv = T_inv @ T

            err_fwd = np.max(np.abs(composed_fwd - np.eye(4)))
            err_inv = np.max(np.abs(composed_inv - np.eye(4)))

            assert err_fwd < 1e-6, f"Batch {b} forward composition error: {err_fwd}"
            assert err_inv < 1e-6, f"Batch {b} inverse composition error: {err_inv}"

    def test_batched_grid_composition_roundtrip_b4(self):
        """Verifies compose_grids with batched displacement fields (B = 4)."""
        shape = (16, 16)
        B = 4
        id_grid = get_identity_grid_torch(shape).repeat(B, 1, 1, 1)

        disp = torch.randn(B, *shape, 2, dtype=torch.float32) * 0.04
        g1 = id_grid + disp

        # Composing any grid with the identity grid must yield the original grid
        g_comp = compose_grids(g1, id_grid)
        assert g_comp.shape == g1.shape

        diff = (g_comp - g1).abs().max().item()
        assert diff < 1e-6, f"Batched grid composition failed: diff={diff}"


class TestAdversarialPrecisionInvariance:
    """Stress tests on float32 vs float64 precision invariance."""

    def test_disp_roundtrip_float32_and_float64(self):
        """Verifies precision preservation in disp_tensor_to_itk <-> disp_itk_to_tensor."""
        shape = (12, 16, 20)
        ref_32 = ants.from_numpy(np.zeros(shape, dtype=np.float32))
        ref_64 = ants.from_numpy(np.zeros(shape, dtype=np.float64))

        # Float32 exact identity
        d32 = torch.randn(1, 20, 16, 12, 3, dtype=torch.float32)
        itk32 = sp.disp_tensor_to_itk(d32, ref_32)
        rec32 = sp.disp_itk_to_tensor(itk32)
        assert rec32.dtype == torch.float32
        assert (rec32 - d32).abs().max().item() == 0.0

        # Float64 precision preservation through ITK domain
        d64 = torch.randn(1, 20, 16, 12, 3, dtype=torch.float64)
        itk64 = sp.disp_tensor_to_itk(d64, ref_64)
        rec64 = sp.disp_itk_to_tensor(itk64)
        err64 = (rec64.to(torch.float64) - d64).abs().max().item()
        assert err64 < 2e-7, f"Float64 roundtrip error exceeded tolerance: {err64}"

    def test_coordinate_grid_precision_float32_and_float64(self):
        """Verifies get_physical_grid_torch and physical_to_normalized_torch precision in float32 and float64."""
        shape = (15, 21, 27)  # Odd dimensions: center voxel is exactly at index (shape - 1) // 2
        spacing = (0.75, 1.25, 2.5)
        origin = (12.34, -56.78, 90.12)
        direction = np.eye(3)

        # Float32
        g_p32 = sp.get_physical_grid_torch(shape, spacing, origin, direction, dtype=torch.float32)
        g_n32 = sp.physical_to_normalized_torch(g_p32, shape, spacing, origin, direction)
        assert g_n32.dtype == torch.float32
        center_32 = g_n32[0, 7, 10, 13]  # center voxel
        assert (center_32 - 0.0).abs().max().item() < 1e-6

        # Float64
        g_p64 = sp.get_physical_grid_torch(shape, spacing, origin, direction, dtype=torch.float64)
        g_n64 = sp.physical_to_normalized_torch(g_p64, shape, spacing, origin, direction)
        assert g_n64.dtype == torch.float64
        center_64 = g_n64[0, 7, 10, 13]  # center voxel
        assert (center_64 - 0.0).abs().max().item() < 1e-14

    def test_affine_conversion_precision_float32_and_float64(self):
        """Verifies grid_to_physical_affine and physical_to_grid_affine roundtrip in float32 and float64."""
        shape = (14, 18, 22)
        spacing = (0.8, 1.1, 1.4)
        origin = (5.0, -10.0, 15.0)

        # Float32
        fi_32 = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin)
        T_32 = np.array([
            [1.02, -0.01, 0.03, 0.10],
            [0.02,  0.99, -0.02, -0.15],
            [-0.01, 0.02, 1.01,  0.20],
            [0.0,   0.0,  0.0,   1.0]
        ], dtype=np.float32)

        M32, t32 = sp.grid_to_physical_affine(T_32, fi_32, fi_32)
        T32_rec = sp.physical_to_grid_affine(M32, t32, fi_32, fi_32)
        err_32 = np.max(np.abs(T_32 - T32_rec))
        assert err_32 < 1e-5, f"Float32 affine error: {err_32}"

        # Float64
        fi_64 = ants.from_numpy(np.zeros(shape, dtype=np.float64), spacing=spacing, origin=origin)
        T_64 = np.array([
            [1.02, -0.01, 0.03, 0.10],
            [0.02,  0.99, -0.02, -0.15],
            [-0.01, 0.02, 1.01,  0.20],
            [0.0,   0.0,  0.0,   1.0]
        ], dtype=np.float64)

        M64, t64 = sp.grid_to_physical_affine(T_64, fi_64, fi_64)
        T64_rec = sp.physical_to_grid_affine(M64, t64, fi_64, fi_64)
        err_64 = np.max(np.abs(T_64 - T64_rec))
        assert err_64 < 1e-7, f"Float64 affine error: {err_64}"

    def test_autograd_scale_precision_float32_and_float64(self):
        """Verifies compute_autograd_physical_scale under float32 and float64."""
        shape_t32 = torch.tensor([16, 24, 32], dtype=torch.float32)
        spacing_t32 = torch.tensor([1.5, 1.0, 0.5], dtype=torch.float32)
        scale_32 = sp.compute_autograd_physical_scale(shape_t32, spacing_t32, dtype=torch.float32)
        assert scale_32.dtype == torch.float32

        shape_t64 = torch.tensor([16, 24, 32], dtype=torch.float64)
        spacing_t64 = torch.tensor([1.5, 1.0, 0.5], dtype=torch.float64)
        scale_64 = sp.compute_autograd_physical_scale(shape_t64, spacing_t64, dtype=torch.float64)
        assert scale_64.dtype == torch.float64

        diff = (scale_64.to(torch.float32) - scale_32).abs().max().item()
        assert diff < 1e-6
