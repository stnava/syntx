"""
tests/test_spatial_roundtrip.py — Exhaustive Roundtrip Invariance Test Suite for syntx.spatial

Verifies exact numerical identity (L_inf < 10^-6) for disp_tensor_to_itk <-> disp_itk_to_tensor
across 2D/3D isotropic and anisotropic domains, arbitrary direction cosines, batched tensors,
disk serialization, and real-world neuroimaging geometries.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import ants

from syntx.spatial import (
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    reverse_components,
    jacobian_determinant,
    deformation_stats,
    image_to_tensor,
    tensor_to_image,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers & Geometries
# ═══════════════════════════════════════════════════════════════════════════════

def _make_orthogonal_matrix(dim=3, seed=42):
    """Generate a valid orthonormal spatial direction matrix with det = +1."""
    rng = np.random.RandomState(seed)
    mat = rng.randn(dim, dim)
    q, r = np.linalg.qr(mat)
    # Ensure positive determinant (proper rotation without reflection)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: Feature Coverage (Primary Roundtrip Capabilities)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialRoundtripFeatureCoverage:
    """Tier 1: Comprehensive feature coverage of tensor <-> ITK displacement field roundtrips."""

    def test_roundtrip_2d_isotropic(self):
        """2D isotropic non-square grid with identity direction matrix."""
        # ITK image: shape (32, 48) -> PyTorch tensor spatial shape (48, 32)
        ref = ants.from_numpy(
            np.zeros((32, 48), dtype=np.float32),
            spacing=(1.0, 1.0),
            origin=(10.0, -5.0),
            direction=np.eye(2)
        )
        disp_orig = torch.randn(1, 48, 32, 2, dtype=torch.float32)

        itk_disp = disp_tensor_to_itk(disp_orig, ref)
        assert isinstance(itk_disp, ants.ANTsImage)
        assert itk_disp.has_components

        disp_rec = disp_itk_to_tensor(itk_disp)
        assert disp_rec.shape == disp_orig.shape
        linf_err = (disp_orig - disp_rec).abs().max().item()
        assert linf_err < 1e-6, f"2D isotropic roundtrip Linf exceeded: {linf_err}"

    def test_roundtrip_2d_anisotropic_rotation(self):
        """2D anisotropic non-square grid with arbitrary 2D in-plane rotation matrix."""
        theta = 0.42
        r2 = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=np.float64)
        ref = ants.from_numpy(
            np.zeros((36, 52), dtype=np.float32),
            spacing=(1.35, 0.75),
            origin=(15.0, -22.5),
            direction=r2
        )
        disp_orig = torch.randn(1, 52, 36, 2, dtype=torch.float32)

        itk_disp = disp_tensor_to_itk(disp_orig, ref)
        disp_rec = disp_itk_to_tensor(itk_disp)

        assert disp_rec.shape == disp_orig.shape
        linf_err = (disp_orig - disp_rec).abs().max().item()
        assert linf_err < 1e-6, f"2D anisotropic rotation Linf exceeded: {linf_err}"

    def test_roundtrip_3d_isotropic_non_cubic(self):
        """3D isotropic non-cubic grid with canonical identity direction cosines."""
        # ITK shape (24, 28, 32) -> PyTorch spatial shape (32, 28, 24)
        ref = ants.from_numpy(
            np.zeros((24, 28, 32), dtype=np.float32),
            spacing=(1.0, 1.0, 1.0),
            origin=(5.0, 15.0, 25.0),
            direction=np.eye(3)
        )
        disp_orig = torch.randn(1, 32, 28, 24, 3, dtype=torch.float32)

        itk_disp = disp_tensor_to_itk(disp_orig, ref)
        disp_rec = disp_itk_to_tensor(itk_disp)

        assert disp_rec.shape == disp_orig.shape
        linf_err = (disp_orig - disp_rec).abs().max().item()
        assert linf_err < 1e-6, f"3D isotropic non-cubic Linf exceeded: {linf_err}"

    def test_roundtrip_3d_anisotropic_arbitrary_direction(self):
        """3D anisotropic grid with arbitrary non-identity orthonormal direction cosines."""
        direction = _make_orthogonal_matrix(3, seed=123)
        ref = ants.from_numpy(
            np.zeros((18, 22, 26), dtype=np.float32),
            spacing=(0.85, 1.45, 2.15),
            origin=(12.0, -34.0, 56.0),
            direction=direction
        )
        disp_orig = torch.randn(1, 26, 22, 18, 3, dtype=torch.float32)

        itk_disp = disp_tensor_to_itk(disp_orig, ref)
        disp_rec = disp_itk_to_tensor(itk_disp)

        assert disp_rec.shape == disp_orig.shape
        linf_err = (disp_orig - disp_rec).abs().max().item()
        assert linf_err < 1e-6, f"3D anisotropic arbitrary direction Linf exceeded: {linf_err}"

    def test_roundtrip_disk_nifti(self):
        """Export displacement field to temporary NIfTI file on disk, read back via file path."""
        direction = _make_orthogonal_matrix(3, seed=999)
        ref = ants.from_numpy(
            np.zeros((20, 24, 28), dtype=np.float32),
            spacing=(1.2, 0.9, 1.8),
            origin=(-10.0, 20.0, -30.0),
            direction=direction
        )
        disp_orig = torch.randn(1, 28, 24, 20, 3, dtype=torch.float32)

        itk_disp = disp_tensor_to_itk(disp_orig, ref)

        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            ants.image_write(itk_disp, tmp_path)
            # disp_itk_to_tensor accepts string file path directly
            disp_rec = disp_itk_to_tensor(tmp_path)
            assert disp_rec.shape == disp_orig.shape
            linf_err = (disp_orig - disp_rec).abs().max().item()
            assert linf_err < 1e-6, f"Disk NIfTI roundtrip Linf exceeded: {linf_err}"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_roundtrip_batched_tensor(self):
        """Batched displacement field (B = 2 and B = 3) roundtripping through list of ANTsImages."""
        ref = ants.from_numpy(
            np.zeros((16, 20, 24), dtype=np.float32),
            spacing=(1.1, 1.1, 1.5),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3)
        )

        for batch_size in (2, 3):
            disp_orig = torch.randn(batch_size, 24, 20, 16, 3, dtype=torch.float32)
            itk_list = disp_tensor_to_itk(disp_orig, ref)

            assert isinstance(itk_list, list), f"Expected list of ANTsImages for B={batch_size}"
            assert len(itk_list) == batch_size

            disp_rec = disp_itk_to_tensor(itk_list)
            assert disp_rec.shape == (batch_size, 24, 20, 16, 3)
            linf_err = (disp_orig - disp_rec).abs().max().item()
            assert linf_err < 1e-6, f"Batched B={batch_size} roundtrip Linf exceeded: {linf_err}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: Boundary, Edge & Corner Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialRoundtripBoundaryCases:
    """Tier 2: Boundary conditions, corner cases, and adversarial input tests."""

    def test_roundtrip_zero_identity_field(self):
        """Zero displacement field (identity mapping) maintains exact 0.0 numerical difference."""
        ref2d = ants.from_numpy(np.zeros((20, 30), dtype=np.float32))
        disp_zero_2d = torch.zeros(1, 30, 20, 2, dtype=torch.float32)
        rec2d = disp_itk_to_tensor(disp_tensor_to_itk(disp_zero_2d, ref2d))
        assert (disp_zero_2d - rec2d).abs().max().item() == 0.0

        ref3d = ants.from_numpy(np.zeros((14, 16, 18), dtype=np.float32))
        disp_zero_3d = torch.zeros(1, 18, 16, 14, 3, dtype=torch.float32)
        rec3d = disp_itk_to_tensor(disp_tensor_to_itk(disp_zero_3d, ref3d))
        assert (disp_zero_3d - rec3d).abs().max().item() == 0.0

    def test_roundtrip_extreme_magnitudes(self):
        """Displacements with large physical offsets (+/- 100mm) and tiny sub-voxel components."""
        ref = ants.from_numpy(np.zeros((16, 16, 16), dtype=np.float32))
        disp = torch.zeros(1, 16, 16, 16, 3, dtype=torch.float32)
        disp[0, 0, 0, 0] = torch.tensor([125.5, -98.75, 450.125])
        disp[0, 8, 8, 8] = torch.tensor([1e-5, -2e-5, 3e-5])
        disp[0, 15, 15, 15] = torch.tensor([-300.0, 250.0, -150.0])

        rec = disp_itk_to_tensor(disp_tensor_to_itk(disp, ref))
        linf_err = (disp - rec).abs().max().item()
        assert linf_err < 1e-6

    def test_roundtrip_unbatched_tensor(self):
        """Input displacement field without batch dimension (*spatial, dim)."""
        ref = ants.from_numpy(np.zeros((20, 25, 30), dtype=np.float32))
        # Unbatched shape: (30, 25, 20, 3)
        disp_unbatched = torch.randn(30, 25, 20, 3, dtype=torch.float32)

        itk_disp = disp_tensor_to_itk(disp_unbatched, ref)
        assert isinstance(itk_disp, ants.ANTsImage)

        disp_rec = disp_itk_to_tensor(itk_disp)
        # disp_itk_to_tensor standardizes to (1, *spatial, dim)
        assert disp_rec.shape == (1, 30, 25, 20, 3)
        linf_err = (disp_unbatched - disp_rec[0]).abs().max().item()
        assert linf_err < 1e-6

    def test_disp_itk_to_tensor_empty_sequence_raises(self):
        """Empty sequence provided to disp_itk_to_tensor must raise ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            disp_itk_to_tensor([])
        with pytest.raises(ValueError, match="Empty"):
            disp_itk_to_tensor(())

    def test_disp_itk_to_tensor_sequence_of_files(self):
        """Sequence of disk file paths correctly stacks into a batched displacement tensor."""
        ref = ants.from_numpy(np.zeros((12, 14, 16), dtype=np.float32))
        d1 = torch.randn(1, 16, 14, 12, 3, dtype=torch.float32)
        d2 = torch.randn(1, 16, 14, 12, 3, dtype=torch.float32)

        itk1 = disp_tensor_to_itk(d1, ref)
        itk2 = disp_tensor_to_itk(d2, ref)

        with tempfile.NamedTemporaryFile(suffix='_1.nii.gz', delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix='_2.nii.gz', delete=False) as f2:
            p1, p2 = f1.name, f2.name

        try:
            ants.image_write(itk1, p1)
            ants.image_write(itk2, p2)

            stacked = disp_itk_to_tensor([p1, p2])
            assert stacked.shape == (2, 16, 14, 12, 3)
            expected = torch.cat([d1, d2], dim=0)
            assert (stacked - expected).abs().max().item() < 1e-6
        finally:
            for p in (p1, p2):
                if os.path.exists(p):
                    os.unlink(p)

    def test_roundtrip_numpy_input(self):
        """Raw NumPy array input disp_np works identically to PyTorch tensor."""
        ref = ants.from_numpy(np.zeros((22, 28), dtype=np.float32), spacing=(1.5, 2.0))
        arr = np.random.randn(1, 28, 22, 2).astype(np.float32)

        itk_img = disp_tensor_to_itk(arr, ref)
        rec_tensor = disp_itk_to_tensor(itk_img)

        assert rec_tensor.shape == (1, 28, 22, 2)
        linf_err = np.max(np.abs(arr - rec_tensor.numpy()))
        assert linf_err < 1e-6

    def test_metadata_preservation(self):
        """Reference image spatial metadata (origin, spacing, direction) is preserved verbatim."""
        direction = _make_orthogonal_matrix(3, seed=777)
        origin = (12.34, -56.78, 90.12)
        spacing = (0.75, 1.25, 2.0)

        ref = ants.from_numpy(
            np.zeros((15, 20, 25), dtype=np.float32),
            spacing=spacing,
            origin=origin,
            direction=direction
        )
        disp = torch.randn(1, 25, 20, 15, 3, dtype=torch.float32)
        itk_disp = disp_tensor_to_itk(disp, ref)

        assert np.allclose(itk_disp.origin, origin, atol=1e-5)
        assert np.allclose(itk_disp.spacing, spacing, atol=1e-5)
        assert np.allclose(itk_disp.direction, direction, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3: Cross-Feature Interactions
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialRoundtripCrossFeatureInteractions:
    """Tier 3: Interactions with component reversal, Jacobian calculations, and resampling."""

    def test_roundtrip_with_reverse_components_invariance(self):
        """reverse_components is strictly self-inverting and matches internal conversion logic."""
        arr = np.random.randn(10, 12, 14, 3).astype(np.float32)
        rev1 = reverse_components(arr)
        rev2 = reverse_components(rev1)
        np.testing.assert_array_equal(arr, rev2)

        # PyTorch tensor component reversal parity
        t = torch.randn(1, 8, 8, 3)
        t_rev = reverse_components(t)
        np.testing.assert_array_almost_equal(t_rev[0, ..., 0], t[0, ..., 2].numpy())
        np.testing.assert_array_almost_equal(t_rev[0, ..., 2], t[0, ..., 0].numpy())

    def test_roundtrip_with_jacobian_evaluation(self):
        """Calculating Jacobian determinant before and after ITK roundtrip yields identical maps."""
        ref = ants.from_numpy(
            np.zeros((20, 24, 28), dtype=np.float32),
            spacing=(1.0, 1.2, 1.5)
        )
        # Create smooth displacement field
        Z, Y, X = 28, 24, 20
        zz, yy, xx = np.meshgrid(
            np.linspace(0, 2 * np.pi, Z),
            np.linspace(0, 2 * np.pi, Y),
            np.linspace(0, 2 * np.pi, X),
            indexing='ij'
        )
        disp_np = np.stack([
            np.sin(zz) * 0.5,
            np.cos(yy) * 0.5,
            np.sin(xx) * 0.5
        ], axis=-1).astype(np.float32)[np.newaxis, ...]

        disp_torch = torch.from_numpy(disp_np)

        # Jacobian on original tensor
        jac1 = jacobian_determinant(disp_torch, ref_image=ref)

        # Convert to ITK and roundtrip back
        itk_disp = disp_tensor_to_itk(disp_torch, ref)
        disp_roundtripped = disp_itk_to_tensor(itk_disp)

        # Jacobian on roundtripped tensor
        jac2 = jacobian_determinant(disp_roundtripped, ref_image=ref)

        linf_jac = np.max(np.abs(jac1 - jac2))
        assert linf_jac < 1e-6, f"Jacobian determinant map diverged across roundtrip: {linf_jac}"

    def test_roundtrip_with_image_resampling(self):
        """Scalar image roundtrip (image_to_tensor <-> tensor_to_image) preserves exact pixel identity."""
        ref = ants.from_numpy(
            np.arange(24 * 30, dtype=np.float32).reshape(24, 30),
            spacing=(1.1, 0.9),
            origin=(5.0, -10.0)
        )
        t_img = image_to_tensor(ref)
        assert t_img.shape == (1, 1, 24, 30)

        rec_img = tensor_to_image(t_img, ref)
        assert rec_img.shape == ref.shape
        np.testing.assert_allclose(rec_img.numpy(), ref.numpy(), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4: Real-World Workload Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialRoundtripRealWorldWorkloads:
    """Tier 4: Realistic clinical imaging geometries and full deformation stats pipelines."""

    def test_real_world_anisotropic_neuroimaging_geometry(self):
        """Simulates clinical anisotropic neuroimaging (sagittal 0.8 x 0.8 x 2.4mm, oblique plane)."""
        # Clinical matrix size: 48 slices, 64 rows, 60 cols (scaled for test execution)
        direction = np.array([
            [0.9986,  0.0524,  0.0000],
            [-0.0519, 0.9882, -0.1444],
            [-0.0076, 0.1442,  0.9895]
        ], dtype=np.float64)
        # Normalize columns to ensure exact orthonormality
        direction = direction / np.linalg.norm(direction, axis=0, keepdims=True)

        ref = ants.from_numpy(
            np.zeros((60, 64, 48), dtype=np.float32),
            spacing=(0.80, 0.80, 2.40),
            origin=(-95.5, -120.0, -45.2),
            direction=direction
        )

        disp_orig = torch.randn(1, 48, 64, 60, 3, dtype=torch.float32) * 2.5
        itk_disp = disp_tensor_to_itk(disp_orig, ref)
        disp_rec = disp_itk_to_tensor(itk_disp)

        assert disp_rec.shape == disp_orig.shape
        linf_err = (disp_orig - disp_rec).abs().max().item()
        assert linf_err < 1e-6, f"Clinical anisotropic roundtrip Linf exceeded: {linf_err}"

    def test_real_world_deformation_stats_pipeline(self):
        """Full end-to-end deformation analysis pipeline: warp -> stats -> roundtrip -> stats verification."""
        ref = ants.from_numpy(
            np.zeros((20, 22, 24), dtype=np.float32),
            spacing=(1.0, 1.0, 1.0)
        )
        disp = torch.randn(1, 24, 22, 20, 3, dtype=torch.float32) * 0.15

        stats_orig = deformation_stats(disp, ref_image=ref)
        itk_disp = disp_tensor_to_itk(disp, ref)
        disp_rec = disp_itk_to_tensor(itk_disp)
        stats_rec = deformation_stats(disp_rec, ref_image=ref)

        assert np.isclose(stats_orig['folding_pct'], stats_rec['folding_pct'], atol=1e-5)
        assert np.isclose(stats_orig['min_j'], stats_rec['min_j'], atol=1e-5)
        assert np.isclose(stats_orig['max_j'], stats_rec['max_j'], atol=1e-5)
        assert np.isclose(stats_orig['mean_displacement'], stats_rec['mean_displacement'], atol=1e-5)
        assert np.isclose(stats_orig['l2_norm'], stats_rec['l2_norm'], atol=1e-5)
