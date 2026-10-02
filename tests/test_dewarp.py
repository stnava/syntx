"""
test_dewarp.py — Unit tests for syntx.dewarp module.
"""

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


def _make_sphere_pair():
    """Create a 3D reference and anatomy sphere pair for testing."""
    shape = (24, 24, 24)
    ref_arr = np.zeros(shape, dtype=np.float32)
    anat_arr = np.zeros(shape, dtype=np.float32)

    zz, yy, xx = np.ogrid[:24, :24, :24]

    # Anatomy: sphere at center (12, 12, 12), radius 6
    anat_dist = np.sqrt((xx - 12) ** 2 + (yy - 12) ** 2 + (zz - 12) ** 2)
    anat_arr[anat_dist <= 6] = 2.0

    # Reference: sphere slightly shifted along Y (distortion) to (12, 14, 12), radius 6
    ref_dist = np.sqrt((xx - 12) ** 2 + (yy - 14) ** 2 + (zz - 12) ** 2)
    ref_arr[ref_dist <= 6] = 1.0

    ref_img = ants.from_numpy(ref_arr, spacing=(2.0, 2.0, 2.0))
    anat_img = ants.from_numpy(anat_arr, spacing=(1.0, 1.0, 1.0))
    mask_img = ants.from_numpy((anat_arr > 0).astype(np.float32), spacing=(1.0, 1.0, 1.0))

    return ref_img, anat_img, mask_img


def test_transformlist_utilities(tmp_path):
    # normalize
    assert normalize_transformlist(None) == []
    assert normalize_transformlist("NA") == []
    assert normalize_transformlist(["NA"]) == []
    assert normalize_transformlist("file.mat") == ["file.mat"]
    assert normalize_transformlist(["file.mat", None, "file2.mat"]) == ["file.mat", "file2.mat"]
    assert normalize_transformlist(["identity"]) == ["identity"]

    # compose
    composed = compose_transformlists(["a.mat"], ["b.nii.gz", "c.mat"])
    assert composed == ["a.mat", "b.nii.gz", "c.mat"]

    # invert
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


def test_rigid_alignment_and_syn_only(tmp_path):
    ref_img, anat_img, mask_img = _make_sphere_pair()

    # Stage 1: Rigid alignment of anatomy to reference grid
    rigid_res = rigid_align_anatomy_to_reference(
        anatomy=anat_img,
        reference=ref_img,
        anatomy_mask=mask_img,
        output_directory=tmp_path / "rigid",
        reg_iterations=[10],
    )

    assert rigid_res.t1_rigid.shape == ref_img.shape
    assert rigid_res.t1_rigid.spacing == ref_img.spacing
    assert len(rigid_res.forward_transformlist) > 0

    # Stage 2: SyNOnly dewarping of reference to anatomy_rigid
    syn_res = syn_only_reference_to_anatomy_rigid(
        reference=ref_img,
        anatomy_rigid=rigid_res.t1_rigid,
        output_directory=tmp_path / "syn",
        restrict_transformation=(0, 1, 0),
        reg_iterations=[10, 5],
    )

    assert syn_res.dewarped.shape == ref_img.shape
    assert len(syn_res.forward_transformlist) > 0
    assert len(syn_res.inverse_transformlist) > 0


def test_dewarp_to_anatomical_pipeline(tmp_path):
    ref_img, anat_img, mask_img = _make_sphere_pair()

    # Create secondary reference (opposite distortion along Y to (12, 10, 12))
    sec_arr = np.zeros(ref_img.shape, dtype=np.float32)
    zz, yy, xx = np.ogrid[:24, :24, :24]
    sec_dist = np.sqrt((xx - 12) ** 2 + (yy - 10) ** 2 + (zz - 12) ** 2)
    sec_arr[sec_dist <= 6] = 1.0
    sec_img = ants.from_numpy(sec_arr, spacing=ref_img.spacing)

    # Run dual dewarping pipeline
    res = dewarp_to_anatomical(
        reference=ref_img,
        anatomy=anat_img,
        anatomy_mask=mask_img,
        secondary_reference=sec_img,
        restrict_transformation=(0, 1, 0),
        output_directory=tmp_path / "dual",
        reg_iterations=[10, 5],
    )

    assert res.dewarped is not None
    assert res.dewarped_primary is not None
    assert res.dewarped_secondary is not None
    assert res.anatomy_rigid is not None
    assert "dewarped_primary_secondary_ncc" in res.metrics

    # Test applying composed transform chains to native anatomy
    app_fwd = apply_transform_chain(
        moving=ref_img,
        fixed=anat_img,
        transformlist=res.transforms_to_anatomy_native,
        whichtoinvert=res.transforms_to_anatomy_native_flags,
    )
    assert app_fwd.warped_image.shape == anat_img.shape

    # Test applying composed transform chains from native anatomy back to native reference
    app_inv = apply_transform_chain(
        moving=mask_img,
        fixed=ref_img,
        transformlist=res.transforms_from_anatomy_native,
        whichtoinvert=res.transforms_from_anatomy_native_flags,
    )
    assert app_inv.warped_image.shape == ref_img.shape


def test_canonical_contract_aliases(tmp_path):
    """Verify legacy parameter names match antsxslowflow canonical_contract."""
    ref_img, anat_img, mask_img = _make_sphere_pair()

    rigid_res = syntx.rigid_align_masked_anatomy_to_motion_reference(
        anatomy_t1=anat_img,
        anatomy_brain_mask=mask_img,
        motion_reference_isotropic=ref_img,
        output_directory=tmp_path / "alias_rigid",
    )
    assert rigid_res.t1_rigid is not None

    syn_res = syntx.syn_only_motion_reference_to_t1_rigid(
        motion_reference_isotropic=ref_img,
        t1_rigid=rigid_res.t1_rigid,
        output_directory=tmp_path / "alias_syn",
        restrict_transformation=(0, 1, 0),
        reg_iterations=[5],
    )
    assert syn_res.motion_reference_in_t1_rigid is not None
