"""Tests of syntx.phase_encode: resolution against headers, vocabulary, conflicts, restriction, relation, round trip.

Physical convention (ITK/ANTs LPS): +x Left, +y Posterior, +z Superior. ``direction`` columns are voxel axes i, j, k.
Expected values below are written by hand from that convention, not computed by the code under test.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

ants = pytest.importorskip("ants")

import syntx
from syntx import phase_encode as pem
from syntx.phase_encode import (
    PE_UNRESTRICTED,
    PhaseEncode,
    PhaseEncodeConflict,
    PhaseEncodeError,
    PhaseEncodeObliqueError,
    PhaseEncodeRelationError,
    PhaseEncodeUndecidable,
    PhaseEncodeVocabularyError,
    parse_pe_label,
    pe_entity_from_path,
    pe_from_dict,
    pe_to_dict,
    relate_phase_encodes,
    resolve_phase_encode,
    restrict_for_image,
    restrict_transformation,
    restrict_with_provenance,
)


def img(direction, shape=(6, 7, 8)):
    d = np.asarray(direction, dtype=float)
    return ants.from_numpy(np.zeros(shape + ((2,) if d.shape[0] == 4 else ()), dtype="float32")
                           if d.shape[0] == 4 else np.zeros(shape[: d.shape[0]], dtype="float32"),
                           direction=d)


def rot_z(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


# name -> direction (columns = voxel axes i,j,k in LPS)
GRIDS = {
    "LPS": np.diag([1.0, 1.0, 1.0]),
    "LAS": np.diag([1.0, -1.0, 1.0]),  # i toward Left, j toward Anterior, k toward Superior
    "RAS": np.diag([-1.0, -1.0, 1.0]),
    "LPI": np.diag([1.0, 1.0, -1.0]),
    "perm_ij": np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]]),  # i -> +y, j -> +x
}
# hand-written expectation: per grid, BIDS label -> (physical_axis, unit vector in LPS)
EXPECT_BIDS = {
    "LPS": {"i": (0, (1, 0, 0)), "i-": (0, (-1, 0, 0)), "j": (1, (0, 1, 0)), "j-": (1, (0, -1, 0)),
            "k": (2, (0, 0, 1)), "k-": (2, (0, 0, -1))},
    "LAS": {"i": (0, (1, 0, 0)), "i-": (0, (-1, 0, 0)), "j": (1, (0, -1, 0)), "j-": (1, (0, 1, 0)),
            "k": (2, (0, 0, 1)), "k-": (2, (0, 0, -1))},
    "RAS": {"i": (0, (-1, 0, 0)), "i-": (0, (1, 0, 0)), "j": (1, (0, -1, 0)), "j-": (1, (0, 1, 0)),
            "k": (2, (0, 0, 1)), "k-": (2, (0, 0, -1))},
    "LPI": {"i": (0, (1, 0, 0)), "i-": (0, (-1, 0, 0)), "j": (1, (0, 1, 0)), "j-": (1, (0, -1, 0)),
            "k": (2, (0, 0, -1)), "k-": (2, (0, 0, 1))},
    "perm_ij": {"i": (1, (0, 1, 0)), "i-": (1, (0, -1, 0)), "j": (0, (1, 0, 0)), "j-": (0, (-1, 0, 0)),
                "k": (2, (0, 0, 1)), "k-": (2, (0, 0, -1))},
}
ANAT_OF_VECTOR = {(1, 0, 0): "RL", (-1, 0, 0): "LR", (0, 1, 0): "AP", (0, -1, 0): "PA", (0, 0, 1): "IS", (0, 0, -1): "SI"}


# ----------------------------------------------------------------------------- grids x labels
@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("label", ["i", "i-", "j", "j-", "k", "k-"])
def test_bids_label_physical_vector_axis_sign(grid, label):
    pe = resolve_phase_encode(img(GRIDS[grid]), label=label)
    axis, vec = EXPECT_BIDS[grid][label]
    assert pe.label == label
    assert pe.physical_axis == axis
    assert np.allclose(pe.physical_vector, vec)
    assert pe.anatomical == ANAT_OF_VECTOR[vec]
    assert pe.alignment == pytest.approx(1.0)
    assert pe.voxel_axis == "ijk".index(label[0]) and pe.sign == (-1 if label.endswith("-") else 1)
    w = restrict_transformation(pe)
    assert w == tuple(1.0 if a == axis else 0.0 for a in range(3))


@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("name", ["AP", "PA", "LR", "RL", "SI", "IS", "dir-AP", "dir-PA"])
def test_named_label_maps_back_to_the_voxel_axis_through_the_header(grid, name):
    bare = name.replace("dir-", "")
    vec = {"AP": (0, 1, 0), "PA": (0, -1, 0), "LR": (-1, 0, 0), "RL": (1, 0, 0), "SI": (0, 0, -1), "IS": (0, 0, 1)}[bare]
    pe = resolve_phase_encode(img(GRIDS[grid]), label=name)
    assert np.allclose(pe.physical_vector, vec)
    assert pe.anatomical == bare
    # the BIDS label it is equivalent to on this header: the one whose expected vector is `vec`
    equivalent = [lab for lab, (_, v) in EXPECT_BIDS[grid].items() if v == vec]
    assert equivalent == [pe.label]


def test_hand_checked_examples():
    # AP acquisition on a RAS nifti is BIDS "j-" (decreasing j runs toward Posterior)
    assert resolve_phase_encode(img(GRIDS["RAS"]), label="AP").label == "j-"
    # "j-" on LAS is AP; on a LPS grid it is PA
    assert resolve_phase_encode(img(GRIDS["LAS"]), label="j-").anatomical == "AP"
    assert resolve_phase_encode(img(GRIDS["LPS"]), label="j-").anatomical == "PA"


def test_oblique_15_passes_45_raises_or_warns():
    d15 = rot_z(15)
    pe = resolve_phase_encode(img(d15), label="j")
    assert pe.alignment == pytest.approx(np.cos(np.deg2rad(15)))
    assert pe.physical_axis == 1 and pe.anatomical == "AP"
    d45 = rot_z(45)
    with pytest.raises(PhaseEncodeObliqueError):
        resolve_phase_encode(img(d45), label="i")
    with pytest.warns(UserWarning, match="oblique"):
        pe = resolve_phase_encode(img(d45), label="i", on_oblique="warn")
    assert pe.anatomical is None  # oblique: no anatomical name claimed
    assert pe.alignment == pytest.approx(np.cos(np.deg2rad(45)))
    # a named label on an oblique grid is also refused (it is not along a voxel axis)
    with pytest.raises(PhaseEncodeObliqueError):
        resolve_phase_encode(img(d45), label="AP")


def test_4d_image_gives_length_3_restriction_and_2d_gives_2():
    d4 = np.eye(4)
    d4[:3, :3] = GRIDS["perm_ij"]
    im4 = img(d4)
    assert im4.dimension == 4
    assert restrict_for_image(im4, label="j") == (1.0, 0.0, 0.0)
    assert syntx.spatial.restriction_from_orientation(im4, bids_phase_encoding_direction="i") == (0.0, 1.0, 0.0)
    im2 = img(np.eye(2))
    assert im2.dimension == 2
    assert restrict_for_image(im2, label="j-") == (0.0, 1.0)
    with pytest.raises(PhaseEncodeError, match="spatial dimensions"):
        resolve_phase_encode(im2, label="k")
    with pytest.raises(PhaseEncodeError):
        resolve_phase_encode(im2, label="SI")


# ----------------------------------------------------------------------------- vocabulary
@pytest.mark.parametrize("bad", ["ROW", "COL", "row", "column", "x", "y", "z", "", " ", "map", "path", "para", "space", "ap_", "jj", "I", "ij", "dir-", "UNK"])
def test_vocabulary_rejections(bad):
    with pytest.raises(PhaseEncodeVocabularyError):
        parse_pe_label(bad)
    with pytest.raises(PhaseEncodeVocabularyError):
        resolve_phase_encode(img(np.eye(3)), label=bad)


def test_dicom_row_col_message_says_not_bids():
    with pytest.raises(PhaseEncodeVocabularyError, match="DICOM"):
        parse_pe_label("COL")


def test_label_none_and_non_string_are_vocabulary_errors():
    with pytest.raises(PhaseEncodeVocabularyError):
        parse_pe_label(None)
    with pytest.raises(PhaseEncodeVocabularyError):
        parse_pe_label(1)


@pytest.mark.parametrize("name,expected", [
    ("sub-01_dir-AP_dwi.nii.gz", "AP"), ("sub-01_acq-x_dir-PA_run-1_epi.nii.gz", "PA"), ("dir-LR_bold.nii.gz", "LR"),
    ("/data/dir-AP/sub-01_map_path_dwi.nii.gz", None),  # dir- in a directory name is not the file's entity
    ("sub-01_map_dwi.nii.gz", None), ("sub-01_path_space-pa_dwi.nii.gz", None), ("sub-01_dir-UNK_dwi.nii.gz", None),
    ("sub-01_capture_para_dwi.nii.gz", None),
])
def test_entity_extraction_is_strict(name, expected):
    assert pe_entity_from_path(name) == expected


# ----------------------------------------------------------------------------- sources, conflicts
def test_sidecar_dict_and_path_agree(tmp_path):
    im = img(GRIDS["LAS"])
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"PhaseEncodingDirection": "j-"}))
    a = resolve_phase_encode(im, sidecar=p)
    b = resolve_phase_encode(im, sidecar={"PhaseEncodingDirection": "j-"})
    assert a.source == "sidecar" and a.source_detail == str(p)
    assert (a.label, a.physical_vector) == (b.label, b.physical_vector)


def test_sources_that_agree_after_the_header_are_accepted_and_recorded():
    # LAS: j- is AP; entity says AP, sidecar says j-: consistent, evidence lists both
    im = img(GRIDS["LAS"])
    pe = resolve_phase_encode(im, sidecar={"PhaseEncodingDirection": "j-"}, entity_path="sub-1_dir-AP_dwi.nii.gz")
    assert pe.source == "sidecar"
    assert [e["source"] for e in pe.evidence["sources"]] == ["sidecar", "entity"]


def test_conflict_entity_vs_sidecar_on_a_grid_where_j_is_x():
    im = img(GRIDS["perm_ij"])  # j is physical x here, so entity AP (y) contradicts sidecar j
    with pytest.raises(PhaseEncodeConflict) as ei:
        resolve_phase_encode(im, sidecar={"PhaseEncodingDirection": "j"}, entity_path="sub-1_dir-AP_dwi.nii.gz")
    assert "sidecar" in str(ei.value) and "entity" in str(ei.value)


def test_conflict_on_sign():
    # LPS: j = +y = AP, so "j" and dir-AP agree while "j-" (PA) contradicts it
    resolve_phase_encode(img(GRIDS["LPS"]), label="j", entity_path="x_dir-AP_dwi.nii.gz")
    with pytest.raises(PhaseEncodeConflict):
        resolve_phase_encode(img(GRIDS["LPS"]), label="j-", entity_path="x_dir-AP_dwi.nii.gz")


def test_undecidable_without_any_source():
    im = img(np.eye(3))
    with pytest.raises(PhaseEncodeUndecidable):
        resolve_phase_encode(im)
    with pytest.raises(PhaseEncodeUndecidable, match="no PhaseEncodingDirection key"):
        resolve_phase_encode(im, sidecar={"RepetitionTime": 2.0})
    with pytest.raises(PhaseEncodeUndecidable):
        resolve_phase_encode(im, entity_path="sub-1_dir-UNK_dwi.nii.gz")


def test_missing_or_corrupt_sidecar_file_is_a_named_error(tmp_path):
    im = img(np.eye(3))
    with pytest.raises(PhaseEncodeError, match="does not exist"):
        resolve_phase_encode(im, sidecar=tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(PhaseEncodeError, match="not readable JSON"):
        resolve_phase_encode(im, sidecar=bad)


def test_sidecar_with_bad_value_is_vocabulary_error_even_if_other_sources_exist():
    with pytest.raises(PhaseEncodeVocabularyError):
        resolve_phase_encode(img(np.eye(3)), sidecar={"PhaseEncodingDirection": "ROW"}, label="j")


def test_inference_hook():
    im = img(GRIDS["perm_ij"])
    with pytest.raises(PhaseEncodeUndecidable, match="inference callable"):
        resolve_phase_encode(im, allow_inference=True)
    pe = resolve_phase_encode(im, allow_inference=True, inference=lambda image: 0)
    assert pe.source == "inferred" and pe.sign is None and pe.label is None
    assert pe.physical_axis == 0 and pe.voxel_axis == 1  # physical x is voxel j on this grid
    assert restrict_transformation(pe) == (1.0, 0.0, 0.0)
    # inference is never used when a source exists
    called = []
    resolve_phase_encode(im, label="i", allow_inference=True, inference=lambda image: called.append(1) or 0)
    assert not called
    with pytest.raises(PhaseEncodeRelationError):
        relate_phase_encodes(pe, pe)


# ----------------------------------------------------------------------------- policy
def test_policy_unrestricted_is_explicit_and_recorded():
    im = img(np.eye(3))
    w, prov = restrict_with_provenance(im, pe_policy="unrestricted")
    assert w is None and prov["policy"] == "unrestricted" and prov["phase_encode"] is None
    assert restrict_for_image(im, pe_policy=PE_UNRESTRICTED) is None
    assert restrict_transformation(PE_UNRESTRICTED) is None
    with pytest.raises(PhaseEncodeConflict):
        restrict_for_image(im, pe_policy="unrestricted", label="j")
    with pytest.raises(ValueError):
        restrict_for_image(im, pe_policy="whatever", label="j")


def test_no_default_axis_in_restrict_for_image():
    with pytest.raises(PhaseEncodeUndecidable):
        restrict_for_image(img(np.eye(3)))
    with pytest.raises(PhaseEncodeUndecidable):
        restrict_for_image(img(np.eye(3)), pe_policy="require")


def test_restriction_provenance_records_the_applied_tuple():
    im = img(GRIDS["perm_ij"])
    w, prov = restrict_with_provenance(im, sidecar={"PhaseEncodingDirection": "j-"})
    assert w == (1.0, 0.0, 0.0)
    assert prov["restrict_transformation"] == [1.0, 0.0, 0.0]
    assert prov["phase_encode"]["label"] == "j-"
    json.dumps(prov)
    # a resolved record (or its dict) can be reused without the sources
    assert restrict_for_image(im, pe=prov["phase_encode"]) == w
    with pytest.raises(PhaseEncodeError):
        restrict_for_image(im, pe=prov["phase_encode"], label="j")


# ----------------------------------------------------------------------------- relation
@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_relate_opposed_same_and_orthogonal(grid):
    im = img(GRIDS[grid])
    j, jm, i = (resolve_phase_encode(im, label=x) for x in ("j", "j-", "i"))
    assert relate_phase_encodes(j, jm) == "opposed"
    assert relate_phase_encodes(j, j) == "same"
    with pytest.raises(PhaseEncodeRelationError):
        relate_phase_encodes(j, i)


def test_relate_across_different_grids_uses_physical_vectors():
    # j on LAS is -y; the same acquisition seen on an LPS grid is "j-"
    a = resolve_phase_encode(img(GRIDS["LAS"]), label="j")
    b = resolve_phase_encode(img(GRIDS["LPS"]), label="j-")
    assert relate_phase_encodes(a, b) == "same"
    assert relate_phase_encodes(a, resolve_phase_encode(img(GRIDS["LPS"]), label="j")) == "opposed"


def test_relate_ambiguous_oblique_pair_raises():
    a = resolve_phase_encode(img(rot_z(0)), label="j")
    # 30 degrees apart but each within tolerance of the (shared) axis when tolerance is loosened
    b = resolve_phase_encode(img(rot_z(30)), label="j", oblique_tol=0.8)
    with pytest.raises(PhaseEncodeRelationError, match="ambiguous"):
        relate_phase_encodes(a, b)  # dot = cos 30 = 0.866 < 0.95


# ----------------------------------------------------------------------------- round trip
@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_dict_roundtrip_is_json_safe_and_exact(grid):
    pe = resolve_phase_encode(img(GRIDS[grid]), sidecar={"PhaseEncodingDirection": "j-"}, entity_path="a_dir-AP_dwi.nii.gz"
                              if grid in ("LAS", "RAS") else None)
    d = pe_to_dict(pe)
    d2 = json.loads(json.dumps(d))
    assert pe_from_dict(d2) == pe
    assert pe_to_dict(pe_from_dict(d2)) == d


def test_from_dict_rejects_malformed():
    with pytest.raises(PhaseEncodeError):
        pe_from_dict({"label": "j"})
    pe = resolve_phase_encode(img(np.eye(3)), label="j")
    d = pe_to_dict(pe)
    d["sign"] = 0
    with pytest.raises(PhaseEncodeError):
        pe_from_dict(d)
    d = pe_to_dict(pe)
    d["schema"] = "other/9"
    with pytest.raises(PhaseEncodeError):
        pe_from_dict(d)


def test_frozen():
    pe = resolve_phase_encode(img(np.eye(3)), label="j")
    with pytest.raises(Exception):
        pe.sign = -1  # type: ignore[misc]


# ----------------------------------------------------------------------------- shim parity and strictness
@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("label", ["i", "j-", "k"])
def test_restriction_from_orientation_shim_matches_central_module(grid, label):
    im = img(GRIDS[grid])
    shim = syntx.spatial.restriction_from_orientation(im, bids_phase_encoding_direction=label)
    assert shim == restrict_for_image(im, label=label)


def test_shim_is_strict_on_bad_input(tmp_path):
    im = img(np.eye(3))
    r = syntx.spatial.restriction_from_orientation
    with pytest.raises(ValueError):
        r(im, bids_phase_encoding_direction="ROW")
    with pytest.raises(PhaseEncodeVocabularyError):
        r(im, bids_phase_encoding_direction="map")
    with pytest.raises(ValueError, match="does not exist"):
        r(im, json_sidecar=tmp_path / "missing.json")
    empty = tmp_path / "e.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="No PhaseEncodingDirection"):
        r(im, json_sidecar=empty)


# ----------------------------------------------------------------------------- known-shift phantom
def _phantom():
    arr = np.zeros((20, 22, 24), dtype="float32")
    arr[5:12, 6:15, 8:18] = 1.0
    arr[7:10, 8:12, 10:14] = 2.0
    arr[9:11, 12:14, 14:17] = 1.5
    return arr


def _centroid_mm(image):
    return np.asarray(ants.get_center_of_mass(image), dtype=float)


def _signed_perm(rng):
    perm = rng.permutation(3)
    d = np.zeros((3, 3))
    for col, row in enumerate(perm):
        d[row, col] = rng.choice([-1.0, 1.0])
    return d


@pytest.mark.parametrize("seed", range(12))
def test_resolved_axis_is_the_axis_of_a_known_pe_shift_under_random_flips_and_permutations(seed):
    """Shift the phantom by +s voxels along array axis a (a synthetic PE shift); resolving label 'ijk'[a]
    must name the physical axis and direction along which the phantom's physical centroid actually moved."""
    rng = np.random.default_rng(seed)
    D = _signed_perm(rng)
    a = int(rng.integers(0, 3))
    sgn = int(rng.choice([-1, 1]))
    spacing = tuple(float(x) for x in rng.uniform(1.0, 2.5, 3))
    arr = _phantom()
    fixed = ants.from_numpy(arr, spacing=spacing, direction=D)
    moved = ants.from_numpy(np.roll(arr, 3 * sgn, axis=a), spacing=spacing, direction=D)
    label = "ijk"[a] + ("-" if sgn < 0 else "")  # a +sgn voxel shift = encoding along sign*axis
    shift_mm = _centroid_mm(moved) - _centroid_mm(fixed)
    pe = resolve_phase_encode(fixed, label=label)
    assert int(np.argmax(np.abs(shift_mm))) == pe.physical_axis
    # off-axis displacement is zero, and the direction of travel matches the resolved vector
    off = np.delete(shift_mm, pe.physical_axis)
    assert np.allclose(off, 0.0, atol=1e-4)
    assert np.dot(shift_mm, pe.physical_vector) > 0
    assert restrict_for_image(fixed, label=label) == tuple(1.0 if k == pe.physical_axis else 0.0 for k in range(3))


def test_syn_with_resolved_restriction_confines_displacement_to_the_pe_axis_on_a_flipped_permuted_grid(tmp_path):
    D = GRIDS["perm_ij"] * np.array([1.0, -1.0, 1.0])  # i -> +y ; j -> -x (flipped and permuted)
    arr = _phantom()
    fixed = ants.from_numpy(arr, spacing=(2.0, 2.0, 2.0), direction=D)
    moving = ants.from_numpy(np.roll(arr, 2, axis=1), spacing=(2.0, 2.0, 2.0), direction=D)  # shift along voxel j
    w = restrict_for_image(fixed, sidecar={"PhaseEncodingDirection": "j"})
    assert w == (1.0, 0.0, 0.0)  # voxel j is physical x on this grid
    res = syntx.syn(fixed, moving, type_of_transform="SyNOnly", reg_iterations=[20, 10], restrict_transformation=w,
                    outprefix=str(tmp_path / "pe_"), verbose=False)
    warp = next(p for p in res["fwdtransforms"] if str(p).endswith(".nii.gz"))
    f = ants.image_read(warp).numpy().reshape(-1, 3)
    mean = np.abs(f).mean(axis=0)
    assert mean[1] == pytest.approx(0.0, abs=1e-5) and mean[2] == pytest.approx(0.0, abs=1e-5)
    assert mean[0] > 0.05


# ----------------------------------------------------------------------------- contract
def test_modality_unit_rejects_partial_pe_lists():
    from syntx.contract import ModalityUnit

    pe = pe_to_dict(resolve_phase_encode(img(np.eye(3)), label="j"))
    ok = ModalityUnit("p", "s", "t", "DTI", "1", ["a.nii.gz", "b.nii.gz"], phase_encode=[pe, pe])
    assert len(ok.phase_encode) == 2
    ModalityUnit("p", "s", "t", "DTI", "1", ["a.nii.gz", "b.nii.gz"])  # empty is allowed (nothing claimed)
    with pytest.raises(ValueError, match="phase_encoding"):
        ModalityUnit("p", "s", "t", "DTI", "1", ["a.nii.gz", "b.nii.gz"], phase_encoding=["j"])
    with pytest.raises(ValueError, match="phase_encode"):
        ModalityUnit("p", "s", "t", "DTI", "1", ["a.nii.gz", "b.nii.gz"], phase_encode=[pe])


# ----------------------------------------------------------------------------- removed silent defaults
def test_dewarp_functions_have_no_default_restriction_and_refuse_none():
    import inspect

    from syntx import dewarp

    for fn in (dewarp.syn_only_reference_to_anatomy_rigid, dewarp.dewarp_to_anatomical):
        p = inspect.signature(fn).parameters["restrict_transformation"]
        assert p.default is inspect.Parameter.empty, f"{fn.__name__} must require restrict_transformation"
    im = img(np.eye(3), shape=(8, 8, 8))
    with pytest.raises(TypeError):
        dewarp.syn_only_reference_to_anatomy_rigid(reference=im, anatomy_rigid=im)
    with pytest.raises(TypeError, match="not accepted"):
        dewarp.syn_only_reference_to_anatomy_rigid(reference=im, anatomy_rigid=im, restrict_transformation=None)
    with pytest.raises(PhaseEncodeVocabularyError):
        dewarp.syn_only_reference_to_anatomy_rigid(reference=im, anatomy_rigid=im, restrict_transformation="y")


def test_dewarp_resolver_accepts_every_form():
    from syntx.dewarp import _resolve_restrict_transformation as res

    im = img(GRIDS["perm_ij"])
    pe = resolve_phase_encode(im, label="j")
    assert res((1, 0, 0), im) == (1.0, 0.0, 0.0)
    assert res("j", im) == (1.0, 0.0, 0.0)
    assert res("PA", im) == (0.0, 1.0, 0.0)  # perm_ij: voxel i is +y
    assert res(pe, im) == (1.0, 0.0, 0.0)
    assert res(pe_to_dict(pe), im) == (1.0, 0.0, 0.0)
    assert res("unrestricted", im) is None
    with pytest.raises(ValueError):
        res((1, 0), im)
    with pytest.raises(ValueError):
        res((1, 2, 0), im)
