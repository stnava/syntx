"""
tests/test_adversarial_spatial_m1.py — Empirical Challenger Stress Suite for syntx.spatial

Adversarially stress-tests the numerical roundtrip invariance of:
    disp_tensor_to_itk <-> disp_itk_to_tensor
under extreme anisotropies, oblique direction cosines, degenerate dimensions,
extreme magnitudes (10^4 to 10^-8), zero fields, multi-batch tensors (B=1,2,4),
and non-standard 2D/3D geometries.
"""

import pytest
import numpy as np
import torch
import ants

from syntx.spatial import (
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    export_ants_displacement_field,
)


def _make_random_rotation(dim=3, seed=None):
    """Generate a random proper rotation matrix in SO(dim)."""
    rng = np.random.RandomState(seed)
    mat = rng.randn(dim, dim)
    q, r = np.linalg.qr(mat)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float64)


class TestAdversarialRoundtripInvariance:
    """Empirical adversarial verification of disp_tensor_to_itk <-> disp_itk_to_tensor."""

    # -------------------------------------------------------------------------
    # 1. High aspect-ratio anisotropic volumes (spacing 0.2 x 1.5 x 4.0 mm, shape 16 x 64 x 128)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("shape_order", ["itk_shape", "tensor_shape"])
    def test_high_aspect_ratio_anisotropy(self, shape_order):
        """Verify high aspect-ratio anisotropy with 20x spacing discrepancy (0.2 to 4.0 mm)."""
        spacing = (0.2, 1.5, 4.0)
        origin = (-45.0, 120.0, -85.5)
        direction = np.eye(3)

        if shape_order == "itk_shape":
            # ITK spatial shape (16, 64, 128) -> Tensor spatial shape (128, 64, 16)
            itk_shape = (16, 64, 128)
            tensor_shape = (1, 128, 64, 16, 3)
        else:
            # Tensor spatial shape (16, 64, 128) -> ITK spatial shape (128, 64, 16)
            itk_shape = (128, 64, 16)
            tensor_shape = (1, 16, 64, 128, 3)

        ref_image = ants.from_numpy(
            np.zeros(itk_shape, dtype=np.float32),
            spacing=spacing,
            origin=origin,
            direction=direction,
        )

        disp_tensor = torch.randn(*tensor_shape, dtype=torch.float32)
        itk_field = disp_tensor_to_itk(disp_tensor, ref_image)
        rec_tensor = disp_itk_to_tensor(itk_field)

        assert rec_tensor.shape == disp_tensor.shape
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err < 1e-6, f"High aspect-ratio anisotropy failed: Linf={linf_err}"
        assert linf_err == 0.0, f"Expected exact bitwise preservation, got Linf={linf_err}"

    # -------------------------------------------------------------------------
    # 2. Arbitrary 3D rotation and oblique direction cosine matrices
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_arbitrary_3d_oblique_direction_cosines(self, seed):
        """Stress-test with arbitrary continuous 3D rotations from SO(3)."""
        rot_matrix = _make_random_rotation(dim=3, seed=seed)
        ref_image = ants.from_numpy(
            np.zeros((18, 22, 26), dtype=np.float32),
            spacing=(0.75, 1.25, 2.5),
            origin=(150.0, -80.0, 35.0),
            direction=rot_matrix,
        )

        disp_tensor = torch.randn(1, 26, 22, 18, 3, dtype=torch.float32)
        itk_field = disp_tensor_to_itk(disp_tensor, ref_image)
        rec_tensor = disp_itk_to_tensor(itk_field)

        assert rec_tensor.shape == disp_tensor.shape
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err < 1e-6, f"Oblique direction matrix (seed {seed}) failed: Linf={linf_err}"
        assert linf_err == 0.0

        # Check metadata preservation on ANTsImage
        np.testing.assert_allclose(itk_field.direction, rot_matrix, atol=1e-6)

    def test_discrete_coordinate_reflection_direction(self):
        """Test with reflection / non-standard radiological coordinate orientation."""
        refl_matrix = np.diag([-1.0, 1.0, 1.0])
        ref_image = ants.from_numpy(
            np.zeros((20, 20, 20), dtype=np.float32),
            direction=refl_matrix,
        )
        disp_tensor = torch.randn(1, 20, 20, 20, 3, dtype=torch.float32)
        rec_tensor = disp_itk_to_tensor(disp_tensor_to_itk(disp_tensor, ref_image))
        assert (disp_tensor - rec_tensor).abs().max().item() == 0.0

    # -------------------------------------------------------------------------
    # 3. Zero displacement fields
    # -------------------------------------------------------------------------
    def test_zero_displacement_field_2d_and_3d(self):
        """Zero displacement field (pure identity mapping) must preserve exact zero."""
        # 2D Zero field
        ref_2d = ants.from_numpy(np.zeros((32, 48), dtype=np.float32), spacing=(1.2, 0.8))
        disp_zero_2d = torch.zeros(1, 48, 32, 2, dtype=torch.float32)
        rec_2d = disp_itk_to_tensor(disp_tensor_to_itk(disp_zero_2d, ref_2d))
        assert (disp_zero_2d - rec_2d).abs().max().item() == 0.0

        # 3D Zero field
        ref_3d = ants.from_numpy(np.zeros((16, 24, 32), dtype=np.float32), spacing=(0.5, 1.0, 2.0))
        disp_zero_3d = torch.zeros(1, 32, 24, 16, 3, dtype=torch.float32)
        rec_3d = disp_itk_to_tensor(disp_tensor_to_itk(disp_zero_3d, ref_3d))
        assert (disp_zero_3d - rec_3d).abs().max().item() == 0.0

    # -------------------------------------------------------------------------
    # 4. Extremely large displacement fields (10^4) and sub-voxel displacements (10^-8)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("magnitude", [1e4, 1e2, 1e-4, 1e-8])
    def test_extreme_magnitude_fields(self, magnitude):
        """Verify roundtrip invariance for extreme magnitude scales."""
        ref_image = ants.from_numpy(np.zeros((16, 16, 16), dtype=np.float32))
        disp_tensor = torch.full((1, 16, 16, 16, 3), magnitude, dtype=torch.float32)

        rec_tensor = disp_itk_to_tensor(disp_tensor_to_itk(disp_tensor, ref_image))
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err < 1e-6, f"Extreme magnitude {magnitude} failed: Linf={linf_err}"
        assert linf_err == 0.0

    def test_mixed_dynamic_range_field(self):
        """Mixed dynamic range displacement field spanning 12 orders of magnitude in one tensor."""
        ref_image = ants.from_numpy(np.zeros((16, 16, 16), dtype=np.float32))
        disp_tensor = torch.zeros(1, 16, 16, 16, 3, dtype=torch.float32)

        # Populate with values across 12 orders of magnitude
        disp_tensor[0, 0, 0, :] = 1e4
        disp_tensor[0, 1, 1, :] = -1e4
        disp_tensor[0, 2, 2, :] = 500.25
        disp_tensor[0, 3, 3, :] = -123.75
        disp_tensor[0, 4, 4, :] = 1.0
        disp_tensor[0, 5, 5, :] = 1e-4
        disp_tensor[0, 6, 6, :] = -1e-4
        disp_tensor[0, 7, 7, :] = 1e-8
        disp_tensor[0, 8, 8, :] = -1e-8

        rec_tensor = disp_itk_to_tensor(disp_tensor_to_itk(disp_tensor, ref_image))
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err < 1e-6, f"Mixed dynamic range failed: Linf={linf_err}"
        assert linf_err == 0.0

    # -------------------------------------------------------------------------
    # 5. Batched tensors with varying batch sizes (B = 1, 2, 4)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batched_tensors_3d(self, batch_size):
        """Stress-test displacement tensors with varying batch sizes B in {1, 2, 4}."""
        ref_image = ants.from_numpy(
            np.zeros((14, 18, 22), dtype=np.float32),
            spacing=(1.1, 1.2, 1.3),
            origin=(-10.0, 20.0, -30.0),
        )
        disp_tensor = torch.randn(batch_size, 22, 18, 14, 3, dtype=torch.float32)
        itk_result = disp_tensor_to_itk(disp_tensor, ref_image)

        if batch_size == 1:
            assert isinstance(itk_result, ants.ANTsImage)
        else:
            assert isinstance(itk_result, list)
            assert len(itk_result) == batch_size
            assert all(isinstance(img, ants.ANTsImage) for img in itk_result)

        rec_tensor = disp_itk_to_tensor(itk_result)
        assert rec_tensor.shape == (batch_size, 22, 18, 14, 3)
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err < 1e-6, f"Batch size B={batch_size} failed: Linf={linf_err}"
        assert linf_err == 0.0

    # -------------------------------------------------------------------------
    # 6. 2D anisotropic images with non-standard origin and rotation
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_2d_anisotropic_oblique(self, batch_size):
        """2D anisotropic grid with non-standard origin, spacing, and in-plane rotation angle."""
        theta = 0.610865  # ~35 degrees
        r2 = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=np.float64)

        ref_image = ants.from_numpy(
            np.zeros((28, 64), dtype=np.float32),
            spacing=(0.45, 2.75),
            origin=(-210.5, 432.1),
            direction=r2,
        )

        disp_tensor = torch.randn(batch_size, 64, 28, 2, dtype=torch.float32)
        itk_result = disp_tensor_to_itk(disp_tensor, ref_image)

        rec_tensor = disp_itk_to_tensor(itk_result)
        assert rec_tensor.shape == (batch_size, 64, 28, 2)
        linf_err = (disp_tensor - rec_tensor).abs().max().item()
        assert linf_err < 1e-6, f"2D anisotropic oblique (B={batch_size}) failed: Linf={linf_err}"
        assert linf_err == 0.0

    # -------------------------------------------------------------------------
    # 7. Additional Adversarial Invariants: Strided Memory & Reverse ITK Roundtrip
    # -------------------------------------------------------------------------
    def test_non_contiguous_sliced_memory(self):
        """Displacement tensor from non-contiguous strided memory slice."""
        ref_image = ants.from_numpy(np.zeros((16, 20, 24), dtype=np.float32))
        large_tensor = torch.randn(2, 32, 40, 48, 3, dtype=torch.float32)
        strided_tensor = large_tensor[1:2, ::2, ::2, ::2, :]
        assert not strided_tensor.is_contiguous()

        rec_tensor = disp_itk_to_tensor(disp_tensor_to_itk(strided_tensor, ref_image))
        assert (strided_tensor - rec_tensor).abs().max().item() == 0.0

    def test_reverse_roundtrip_itk_to_tensor_to_itk(self):
        """Starting from native ANTsImage displacement field, convert to tensor then back to ITK."""
        arr_xyz = np.random.randn(20, 24, 28, 3).astype(np.float32)
        orig_itk = ants.from_numpy(
            arr_xyz,
            spacing=(0.6, 1.1, 2.2),
            origin=(-50.0, 75.0, -100.0),
            direction=_make_random_rotation(3, seed=789),
            has_components=True,
        )

        tensor = disp_itk_to_tensor(orig_itk)
        recovered_itk = disp_tensor_to_itk(tensor, orig_itk)

        diff_voxels = np.max(np.abs(orig_itk.numpy() - recovered_itk.numpy()))
        assert diff_voxels == 0.0
        np.testing.assert_allclose(orig_itk.origin, recovered_itk.origin)
        np.testing.assert_allclose(orig_itk.spacing, recovered_itk.spacing)
        np.testing.assert_allclose(orig_itk.direction, recovered_itk.direction)
