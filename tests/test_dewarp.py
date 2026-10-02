"""
test_dewarp.py — Unit tests for syntx.dewarp with small, fast simulated data and known deformations.
"""

from pathlib import Path
import numpy as np
import pytest

ants = pytest.importorskip("ants")
torch = pytest.importorskip("torch")

import syntx
from syntx.dewarp import (
    normalize_transformlist,
    compose_transformlists,
    invert_transformlist,
    infer_inverse_warp_path,
    rigid_align_anatomy_to_reference,
    syn_only_reference_to_anatomy_rigid,
    dewarp_to_anatomical,
    compose_native_to_anatomy_transform,
    compose_anatomy_to_native_frame_transform,
    apply_transform_chain,
)


def _make_asymmetric_phantom(shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0)):
    """Create a small, asymmetric 3D ground-truth phantom with known features.

    Designed with no axis-swap symmetry so coordinate transpositions or sign flips
    cannot hide during registration.
    """
    arr = np.zeros(shape, dtype=np.float32)
    # Outer asymmetric block
    arr[6:16, 6:18, 8:16] = 1.0
    # Nested internal nucleus (eccentric)
    arr[9:13, 9:13, 10:14] = 2.0
    # Additional off-center marker along Z
    arr[7:9, 14:17, 12:15] = 1.5

    gt_img = ants.from_numpy(arr, spacing=spacing)
    mask_arr = (arr > 0).astype(np.float32)
    mask_img = ants.from_numpy(mask_arr, spacing=spacing)
    return gt_img, mask_img, arr


def test_transformlist_utilities(tmp_path):
    """Test transform list parsing, composition, and inverse file mapping."""
    # 1. Normalization
    assert normalize_transformlist(None) == []
    assert normalize_transformlist("NA") == []
    assert normalize_transformlist(["NA"]) == []
    assert normalize_transformlist("file.mat") == ["file.mat"]
    assert normalize_transformlist(["file.mat", None, "file2.mat"]) == ["file.mat", "file2.mat"]
    assert normalize_transformlist(["identity"]) == ["identity"]

    # 2. Composition
    composed = compose_transformlists(["a.mat"], ["b.nii.gz", "c.mat"])
    assert composed == ["a.mat", "b.nii.gz", "c.mat"]

    # 3. Inversion
    fwd_warp = tmp_path / "test1Warp.nii.gz"
    fwd_warp.touch()
    inv_warp = tmp_path / "test1InverseWarp.nii.gz"
    inv_warp.touch()
    affine_mat = tmp_path / "test0GenericAffine.mat"
    affine_mat.touch()

    chain = [str(fwd_warp), str(affine_mat)]
    inv_list, inv_flags = invert_transformlist(chain)

    # Inverted order: affine first with flag True, then inverse warp with flag False
    assert inv_list == [str(affine_mat), str(inv_warp)]
    assert inv_flags == [True, False]


def test_rigid_alignment_recovers_known_displacement(tmp_path):
    """Verify rigid alignment recovers a known integer voxel translation."""
    gt_img, mask_img, gt_arr = _make_asymmetric_phantom(shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0))

    # Reference is ground truth; Anatomy is translated by +2 voxels in X (array axis 0)
    # and -1 voxel in Z (array axis 2)
    known_shift_x = 2
    known_shift_z = -1
    anat_arr = np.roll(np.roll(gt_arr, shift=known_shift_x, axis=0), shift=known_shift_z, axis=2)
    anat_img = ants.from_numpy(anat_arr, spacing=gt_img.spacing)
    anat_mask = ants.from_numpy((anat_arr > 0).astype(np.float32), spacing=gt_img.spacing)

    # Rigid alignment of anatomy to reference grid
    rigid_res = rigid_align_anatomy_to_reference(
        anatomy=anat_img,
        reference=gt_img,
        anatomy_mask=anat_mask,
        output_directory=tmp_path / "rigid",
        reg_iterations=[10],
    )

    # Assert shape and spacing match reference
    assert rigid_res.t1_rigid.shape == gt_img.shape
    assert rigid_res.t1_rigid.spacing == gt_img.spacing
    assert len(rigid_res.forward_transformlist) > 0

    # Assert known displacement was recovered (high correlation with true GT)
    corr = np.corrcoef(rigid_res.t1_rigid.numpy().ravel(), gt_arr.ravel())[0, 1]
    assert corr > 0.90, f"Expected rigid alignment correlation > 0.90, got {corr:.4f}"


def test_syn_only_pe_restriction_on_known_shift(tmp_path):
    """Verify SyNOnly with restrict_transformation strictly confines deformation to PE axis.

    This provides mechanism-level debugging proof that non-PE axes remain exactly zero.
    """
    gt_img, _, gt_arr = _make_asymmetric_phantom(shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0))

    # Shift moving image by +2 voxels strictly along array axis 1 (physical Y / PE axis)
    shift_pe_voxels = 2
    moving_arr = np.roll(gt_arr, shift=shift_pe_voxels, axis=1)
    moving_img = ants.from_numpy(moving_arr, spacing=gt_img.spacing)

    # Register moving to fixed ground truth, restricted to physical axis 1 (Y)
    syn_res = syn_only_reference_to_anatomy_rigid(
        reference=moving_img,
        anatomy_rigid=gt_img,
        output_directory=tmp_path / "syn_pe",
        restrict_transformation=(0, 1, 0),
        reg_iterations=[20, 10],
        verbose=False,
    )

    # 1. Extract warp field and check per-axis displacement magnitudes
    warp_files = [p for p in syn_res.forward_transformlist if str(p).endswith(".nii.gz")]
    assert warp_files, "Expected forward displacement field among forward_transformlist"

    field = ants.image_read(warp_files[0]).numpy()
    field = field.reshape(24, 24, 24, -1)[..., :3]
    mean_disp_xyz = np.abs(field).reshape(-1, 3).mean(axis=0)

    # Physical axis 0 (X) and physical axis 2 (Z) must have strictly ZERO displacement
    assert mean_disp_xyz[0] == pytest.approx(0.0, abs=1e-5), f"Non-zero X disp: {mean_disp_xyz[0]}"
    assert mean_disp_xyz[2] == pytest.approx(0.0, abs=1e-5), f"Non-zero Z disp: {mean_disp_xyz[2]}"

    # Physical axis 1 (Y / PE) must carry the deformation
    assert mean_disp_xyz[1] > 0.1, f"Expected non-zero PE displacement along Y, got {mean_disp_xyz[1]}"

    # 2. Check that dewarped image restored the ground truth
    corr = np.corrcoef(syn_res.dewarped.numpy().ravel(), gt_arr.ravel())[0, 1]
    raw_corr = np.corrcoef(moving_arr.ravel(), gt_arr.ravel())[0, 1]
    assert corr > raw_corr, f"Dewarped corr {corr:.4f} should exceed raw corr {raw_corr:.4f}"
    assert corr > 0.85, f"Expected dewarped correlation > 0.85, got {corr:.4f}"


def test_hybrid_dual_pe_recovers_ground_truth(tmp_path):
    """Test the complete hybrid dual-PE + T1 deformation pipeline with known equal-and-opposite shifts."""
    gt_img, mask_img, gt_arr = _make_asymmetric_phantom(shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0))

    # Distorted AP: +2 voxels along Y (+4.0 mm)
    # Distorted PA: -2 voxels along Y (-4.0 mm)
    ap_arr = np.roll(gt_arr, shift=2, axis=1)
    pa_arr = np.roll(gt_arr, shift=-2, axis=1)

    b0_ap = ants.from_numpy(ap_arr, spacing=gt_img.spacing)
    b0_pa = ants.from_numpy(pa_arr, spacing=gt_img.spacing)

    # Simulated T1 anatomy: slight rigid offset (+1 voxel in X)
    anat_arr = np.roll(gt_arr, shift=1, axis=0)
    anat_img = ants.from_numpy(anat_arr, spacing=gt_img.spacing)
    anat_mask = ants.from_numpy((anat_arr > 0).astype(np.float32), spacing=gt_img.spacing)

    # Run hybrid dual-PE dewarping
    res = dewarp_to_anatomical(
        reference=b0_ap,
        anatomy=anat_img,
        anatomy_mask=anat_mask,
        secondary_reference=b0_pa,
        strategy="hybrid",
        restrict_transformation=(0, 1, 0),
        output_directory=tmp_path / "hybrid",
        reg_iterations=[20, 10],
        verbose=False,
    )

    assert res.dewarped is not None
    assert res.dewarped_primary is not None
    assert res.dewarped_secondary is not None

    # Check that AP-PA consistency improved substantially from raw
    assert res.metrics["dewarped_primary_secondary_ncc"] > res.metrics["raw_primary_secondary_ncc"]

    # Check recovery of true ground truth
    corr_gt = np.corrcoef(res.dewarped.numpy().ravel(), gt_arr.ravel())[0, 1]
    assert corr_gt > 0.85, f"Expected recovery correlation with ground truth > 0.85, got {corr_gt:.4f}"

    # Verify composed single-interpolation transform chains:
    # 1. Raw AP directly into native anatomy space
    app_to_anat = apply_transform_chain(
        moving=b0_ap,
        fixed=anat_img,
        transformlist=res.transforms_to_anatomy_native,
        whichtoinvert=res.transforms_to_anatomy_native_flags,
    )
    assert app_to_anat.warped_image.shape == anat_img.shape

    # 2. Native anatomy back to distorted AP space
    app_from_anat = apply_transform_chain(
        moving=anat_mask,
        fixed=b0_ap,
        transformlist=res.transforms_from_anatomy_native,
        whichtoinvert=res.transforms_from_anatomy_native_flags,
    )
    assert app_from_anat.warped_image.shape == b0_ap.shape


def test_single_pe_dewarp_to_anatomical(tmp_path):
    """Test single-PE path when secondary_reference is None."""
    gt_img, mask_img, gt_arr = _make_asymmetric_phantom(shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0))

    # Moving reference with known PE shift
    ref_arr = np.roll(gt_arr, shift=2, axis=1)
    ref_img = ants.from_numpy(ref_arr, spacing=gt_img.spacing)

    res = dewarp_to_anatomical(
        reference=ref_img,
        anatomy=gt_img,
        anatomy_mask=mask_img,
        secondary_reference=None,
        restrict_transformation=(0, 1, 0),
        output_directory=tmp_path / "single_pe",
        reg_iterations=[15, 5],
        verbose=False,
    )

    assert res.dewarped_secondary is None
    assert res.dewarped is not None
    corr = np.corrcoef(res.dewarped.numpy().ravel(), gt_arr.ravel())[0, 1]
    assert corr > 0.85


def test_canonical_contract_slowflow_compatibility(tmp_path):
    """Verify antsxslowflow canonical_contract wrapper functions work on simulated data."""
    gt_img, mask_img, _ = _make_asymmetric_phantom(shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0))

    rigid_res = syntx.rigid_align_masked_anatomy_to_motion_reference(
        anatomy_t1=gt_img,
        anatomy_brain_mask=mask_img,
        motion_reference_isotropic=gt_img,
        output_directory=tmp_path / "slowflow_rigid",
    )
    assert rigid_res.t1_rigid is not None

    syn_res = syntx.syn_only_motion_reference_to_t1_rigid(
        motion_reference_isotropic=gt_img,
        t1_rigid=rigid_res.t1_rigid,
        output_directory=tmp_path / "slowflow_syn",
        restrict_transformation=(0, 1, 0),
        reg_iterations=[5],
    )
    assert syn_res.motion_reference_in_t1_rigid is not None
