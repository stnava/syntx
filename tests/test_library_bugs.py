"""Unit tests for verified core library bug fixes (Milestone 1: Bugs L1-L4).

- Bug L1: Demean source and destination coordinates in Sinkhorn matching before
  computing spatial distance cost to prevent scanner table z-offsets from overflowing dustbin.
- Bug L2: Add anatomical cranial lower bounds (min >= 110 mm, max >= 140 mm) in diagnose.py
  to prevent small FOV non-brain volumes from being misclassified as brain.
- Bug L3: Slice 4D ANTsImage inputs to 3D in auto_reg and avoid applying thorax HU windowing
  to non-thorax fixed targets.
- Bug L4: Apply anisotropy-safe working-image capping on large sub-millimeter CTs during
  registration while evaluating all final transforms on native full-resolution images.
"""

import os
import pytest
import numpy as np
import torch
import ants

import syntx
from syntx.landmarks.optimal_transport import sinkhorn_matching, weighted_procrustes
from syntx.diagnose import diagnose_image, ImageDiagnosis, PairDiagnosis
from syntx.policy import synthesize_policy, RegistrationPolicy
from syntx.syn import auto_reg
from syntx.robust_affine import _cap_anisotropic_working_image, robust_affine


# ==============================================================================
# Bug L1: Sinkhorn Matching Coordinate Demeaning
# ==============================================================================

class TestBugL1SinkhornDemeaning:
    """Tests for spatial distance demeaning in sinkhorn_matching."""

    def test_sinkhorn_demeaning_recovers_large_table_offset(self):
        """Verify that demeaning coordinates prevents 500mm z-offset from routing points to dustbin."""
        torch.manual_seed(42)
        n_points = 40
        d_feat = 16

        feat_src = torch.randn(n_points, d_feat)
        feat_dst = feat_src.clone()

        # Coords in LPS (mm) centered around origin
        coords_src = torch.randn(n_points, 3) * 20.0
        # Add 500 mm scanner table z-offset to destination
        coords_dst = coords_src.clone()
        coords_dst[:, 2] += 500.0

        # With demeaning (current implementation)
        P_match, weights = sinkhorn_matching(
            feat_src, feat_dst,
            coords_src=coords_src, coords_dst=coords_dst,
            spatial_weight=0.5, dustbin_cost=1.0, epsilon=0.05, n_iters=50
        )

        # In-distribution points should have high matched weight (>0.5)
        high_conf_matches = (weights > 0.5).sum().item()
        assert high_conf_matches == n_points, f"Expected {n_points} high-confidence matches, got {high_conf_matches}"

        # Weighted Procrustes on original coordinates should accurately recover the 500mm translation
        R, t, s, transformed = weighted_procrustes(coords_src, coords_dst, weights=weights)
        assert np.isclose(float(t[2]), 500.0, atol=1.0), f"Expected z translation ~500mm, got {float(t[2]):.2f}mm"
        assert np.isclose(float(s), 1.0, atol=1e-3)
        assert np.allclose(R.numpy(), np.eye(3), atol=1e-2)

    def test_sinkhorn_without_spatial_coordinates(self):
        """Verify that sinkhorn_matching functions without spatial coordinates."""
        feat_src = torch.randn(20, 8)
        feat_dst = torch.randn(25, 8)

        P, w = sinkhorn_matching(feat_src, feat_dst, spatial_weight=0.0)
        assert P.shape == (20, 25)
        assert w.shape == (20,)
        assert torch.all(w >= 0.0) and torch.all(w <= 1.05)

    def test_sinkhorn_with_empty_coords(self):
        """Verify edge case with empty coordinate tensors."""
        feat_src = torch.zeros((0, 8))
        feat_dst = torch.zeros((0, 8))
        coords_src = torch.zeros((0, 3))
        coords_dst = torch.zeros((0, 3))

        P, w = sinkhorn_matching(feat_src, feat_dst, coords_src, coords_dst, spatial_weight=0.5)
        assert P.shape == (0, 0)
        assert w.shape == (0,)


# ==============================================================================
# Bug L2: Anatomical Cranial Lower Bounds in diagnose.py
# ==============================================================================

class TestBugL2CranialLowerBounds:
    """Tests for cranial bounding box criteria in image diagnosis."""

    def test_whole_brain_mri_classified_as_brain(self):
        """Standard whole-brain MRI field of view (min>=110mm, max in [140, 260)mm) -> BRAIN."""
        # 180mm x 200mm x 150mm FOV
        arr = np.random.uniform(0, 1000, size=(180, 200, 150)).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0))

        diag = diagnose_image(img, fast=True)
        assert diag.body_part == "BRAIN", f"Expected BRAIN, got {diag.body_part}"
        assert diag.is_brain()

    def test_small_fov_prostate_slab_not_classified_as_brain(self):
        """Small FOV slab (e.g. Prostate: 192 x 192 x 60 mm; min extent 60 < 110 mm) -> UNKNOWN/not BRAIN."""
        # Spacing (0.6, 0.6, 4.0), shape (320, 320, 15) -> extent (192, 192, 60) mm
        arr = np.random.uniform(0, 500, size=(32, 32, 15)).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(6.0, 6.0, 4.0))  # Extent: 192 x 192 x 60 mm

        diag = diagnose_image(img, fast=True)
        assert diag.body_part != "BRAIN", f"Small FOV slab misclassified as {diag.body_part}"
        assert not diag.is_brain()

    def test_small_fov_hippocampus_crop_not_classified_as_brain(self):
        """Small FOV subcortical crop (e.g. Hippocampus: 35 x 51 x 35 mm) -> UNKNOWN/not BRAIN."""
        arr = np.random.uniform(0, 1000, size=(35, 51, 35)).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0))

        diag = diagnose_image(img, fast=True)
        assert diag.body_part != "BRAIN", f"Hippocampus crop misclassified as {diag.body_part}"
        assert not diag.is_brain()

    def test_small_fov_ct_not_classified_as_brain(self):
        """Small FOV CT with high bone/soft tissue fraction but small extent -> ABDOMEN/UNKNOWN, not BRAIN."""
        # Create CT array with bone (>300 HU) and soft tissue (20-80 HU) but small z-extent (40 mm)
        arr = np.full((120, 120, 20), -1000.0, dtype=np.float32)
        arr[20:100, 20:100, :] = 50.0   # Soft tissue
        arr[10:20, 10:110, :] = 500.0   # Bone rim
        arr[100:110, 10:110, :] = 500.0  # Bone rim
        arr[10:110, 10:20, :] = 500.0   # Bone rim
        arr[10:110, 100:110, :] = 500.0  # Bone rim

        img = ants.from_numpy(arr, spacing=(1.0, 1.0, 2.0))  # Extent: 120 x 120 x 40 mm (min=40 < 110)
        diag = diagnose_image(img, fast=True)
        assert diag.body_part != "BRAIN", f"Small FOV CT misclassified as {diag.body_part}"

    def test_2d_and_4d_brain_classification(self):
        """2D slice or 4D multi-contrast brain images are properly classified."""
        # 2D slice
        arr2d = np.random.uniform(0, 1000, size=(100, 100)).astype(np.float32)
        img2d = ants.from_numpy(arr2d, spacing=(1.0, 1.0))
        assert diagnose_image(img2d, fast=True).body_part == "BRAIN"

        # 4D multi-contrast (4 channels)
        arr4d = np.random.uniform(0, 1000, size=(40, 40, 40, 4)).astype(np.float32)
        img4d = ants.from_numpy(arr4d, spacing=(1.0, 1.0, 1.0, 1.0))
        assert diagnose_image(img4d, fast=True).body_part == "BRAIN"


# ==============================================================================
# Bug L3: 4D auto_reg Slicing & Thorax Window Gating
# ==============================================================================

class TestBugL3AutoReg4DSlicingAndThoraxGating:
    """Tests for 4D image slicing in auto_reg and thorax HU window target gating."""

    def test_auto_reg_slices_4d_image_to_3d(self):
        """Verify that passing 4D ANTsImage to auto_reg slices to 3D and registers successfully."""
        arr4d = np.random.uniform(0, 100, size=(24, 24, 16, 4)).astype(np.float32)
        img4d = ants.from_numpy(arr4d, spacing=(1.5, 1.5, 2.0, 1.0))

        # auto_reg with Affine transform and fast iterations
        with pytest.warns(UserWarning, match="4D fixed image .* slicing axis=3, idx=0 to 3D"):
            res = auto_reg(img4d, img4d, type_of_transform='Affine', reg_iterations=[3], diagnose=False)

        assert res['warpedmovout'].dimension == 3
        assert res['warpedfixout'].dimension == 3
        assert res['warpedmovout'].shape == (24, 24, 16)
        assert 'fwdtransforms' in res
        assert len(res['fwdtransforms']) >= 1

    def test_auto_reg_slices_4d_labels_to_3d(self):
        """Verify that 4D label images are also sliced to 3D for metric evaluation."""
        arr4d = np.random.uniform(0, 100, size=(24, 24, 16, 2)).astype(np.float32)
        lbl4d = np.zeros((24, 24, 16, 2), dtype=np.float32)
        lbl4d[6:18, 6:18, 4:12, :] = 1.0

        img4d = ants.from_numpy(arr4d, spacing=(1.0, 1.0, 2.0, 1.0))
        lbl4d_img = ants.from_numpy(lbl4d, spacing=(1.0, 1.0, 2.0, 1.0))

        with pytest.warns(UserWarning, match="4D .* slicing axis=3, idx=0 to 3D"):
            res = auto_reg(
                img4d, img4d, fixed_label=lbl4d_img, moving_label=lbl4d_img,
                type_of_transform='Affine', reg_iterations=[3], diagnose=False
            )

        assert 'dice_symmetric' in res
        assert res['dice_symmetric'] >= 0.90

    def test_policy_thorax_gating_requires_fixed_thorax(self):
        """Verify that thorax CT policy is only selected when the fixed image is thorax."""
        diag_abd = ImageDiagnosis('CT', 'ABDOMEN', 'HOUNSFIELD', -1000.0, 1000.0, 50.0, (200, 200, 100), 3, 0.9, 'test')
        diag_thx = ImageDiagnosis('CT', 'THORAX', 'HOUNSFIELD', -1000.0, 1000.0, -400.0, (200, 200, 100), 3, 0.9, 'test')

        # Fixed ABDOMEN, Moving THORAX -> must NOT apply thorax lung window [-1000, 400]
        pair_abd_fixed = PairDiagnosis(diag_abd, diag_thx, 'INTRA_SUBJECT', True, 0.9)
        policy_abd = synthesize_policy(pair_abd_fixed)
        assert policy_abd.ct_window == (-150.0, 250.0), f"Expected soft-tissue window, got {policy_abd.ct_window}"
        assert policy_abd.transform_type == "SyN"

        # Fixed THORAX, Moving THORAX -> applies thorax lung window [-1000, 400] and TVF
        pair_thx_fixed = PairDiagnosis(diag_thx, diag_thx, 'INTRA_SUBJECT', True, 0.9)
        policy_thx = synthesize_policy(pair_thx_fixed)
        assert policy_thx.ct_window == (-1000.0, 400.0)
        assert policy_thx.transform_type == "TVF"


# ==============================================================================
# Bug L4: Anisotropy-Safe Working Image Capping in robust_affine
# ==============================================================================

class TestBugL4AnisotropyWorkingImageCapping:
    """Tests for anisotropy-safe working-image capping on large sub-mm CTs."""

    def test_cap_anisotropic_working_image_preserves_thick_slices(self):
        """Verify that coarsening in-plane resolution to 1.0mm does not upsample thick z-slices."""
        # 50 x 50 in-plane with 0.6mm spacing, 12 slices with 5.0mm spacing
        arr = np.random.uniform(-1000, 1000, size=(50, 50, 12)).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(0.6, 0.6, 5.0), origin=(10.0, 20.0, 30.0))

        capped = _cap_anisotropic_working_image(img, max_resolution_mm=1.0)

        # z-axis slice count must be preserved exactly
        assert capped.shape[2] == 12, f"Expected 12 slices, got {capped.shape[2]}"
        assert np.isclose(capped.spacing[2], 5.0, atol=1e-4)

        # in-plane resolution must be coarsened to ~1.0mm (fewer voxels)
        assert capped.shape[0] < 50
        assert capped.shape[1] < 50
        assert np.isclose(capped.spacing[0], 1.0, atol=0.05)
        assert np.isclose(capped.spacing[1], 1.0, atol=0.05)

        # Origin and direction must be preserved
        assert np.allclose(capped.origin, img.origin)
        assert np.allclose(capped.direction, img.direction)

    def test_cap_anisotropic_working_image_noop_for_coarse_image(self):
        """Images already >= max_resolution_mm on all axes must remain unchanged."""
        arr = np.random.uniform(0, 100, size=(30, 30, 30)).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.5, 1.5, 2.0))

        capped = _cap_anisotropic_working_image(img, max_resolution_mm=1.0)
        assert capped is img

    def test_robust_affine_evaluates_on_native_full_resolution(self):
        """Verify that robust_affine optimizes on capped image but returns warpedmovout at native resolution."""
        arr_f = np.random.uniform(-1000, 1000, size=(48, 48, 10)).astype(np.float32)
        arr_m = np.random.uniform(-1000, 1000, size=(48, 48, 10)).astype(np.float32)

        fixed = ants.from_numpy(arr_f, spacing=(0.7, 0.7, 4.0), origin=(5.0, 10.0, 15.0))
        moving = ants.from_numpy(arr_m, spacing=(0.7, 0.7, 4.0), origin=(5.0, 10.0, 15.0))

        res = robust_affine(fixed, moving, mode='auto', iterations=[5])

        # Final output images must match native full-resolution fixed geometry
        assert res['warpedmovout'].shape == fixed.shape
        assert np.allclose(res['warpedmovout'].spacing, fixed.spacing)
        assert np.allclose(res['warpedmovout'].origin, fixed.origin)
        assert np.allclose(res['warpedmovout'].direction, fixed.direction)

        assert res['warpedfixout'].shape == fixed.shape
        assert np.allclose(res['warpedfixout'].spacing, fixed.spacing)

