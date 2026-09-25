"""Unit tests for syntx.spatial.restriction_from_orientation."""
import json
import numpy as np
import pytest

ants = pytest.importorskip("ants")

from syntx.spatial import restriction_from_orientation


def _image_with_direction(direction):
    vol = np.zeros((6, 6, 6), dtype=float)
    return ants.from_numpy(vol, direction=np.asarray(direction, dtype=float))


IDENTITY = np.eye(3)


def test_anatomical_axis_labels_are_direct_world_axes():
    img = _image_with_direction(IDENTITY)
    assert restriction_from_orientation(img, anatomical_axis="LR") == (1.0, 0.0, 0.0)
    assert restriction_from_orientation(img, anatomical_axis="AP") == (0.0, 1.0, 0.0)
    assert restriction_from_orientation(img, anatomical_axis="SI") == (0.0, 0.0, 1.0)
    # case-insensitive, and RL/PA/IS synonyms resolve to the same axis as LR/AP/SI
    assert restriction_from_orientation(img, anatomical_axis="rl") == restriction_from_orientation(img, anatomical_axis="LR")


def test_anatomical_axis_unknown_label_raises():
    img = _image_with_direction(IDENTITY)
    with pytest.raises(ValueError, match="Unknown anatomical_axis"):
        restriction_from_orientation(img, anatomical_axis="XYZZY")


def test_bids_voxel_axis_resolves_through_identity_direction():
    """With an identity direction matrix, BIDS voxel axis j == physical axis 1 (Y)."""
    img = _image_with_direction(IDENTITY)
    w = restriction_from_orientation(img, bids_phase_encoding_direction="j")
    assert w == (0.0, 1.0, 0.0)
    # trailing sign is irrelevant to a restriction weight
    assert restriction_from_orientation(img, bids_phase_encoding_direction="j-") == w


def test_bids_voxel_axis_cross_checked_against_direction_matrix():
    """A permuted (axis-swapped) direction matrix must change which physical axis 'j' maps to."""
    # image axis 1 (j) points along physical X instead of Y
    permuted = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    img = _image_with_direction(permuted)
    w = restriction_from_orientation(img, bids_phase_encoding_direction="j")
    assert w == (1.0, 0.0, 0.0), "j should resolve to physical X under this permuted direction matrix"


def test_oblique_acquisition_warns_but_still_resolves():
    theta = np.deg2rad(30)  # well beyond the ~11.5 deg default threshold
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    img = _image_with_direction(R)
    with pytest.warns(UserWarning, match="obliquely acquired"):
        w = restriction_from_orientation(img, bids_phase_encoding_direction="i")
    # still resolves to *a* physical axis (the dominant one), just with a warning
    assert sum(w) == 1.0 and w.count(1.0) == 1


def test_near_axis_aligned_acquisition_does_not_warn():
    theta = np.deg2rad(2)  # within the default 0.98 alignment threshold
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    img = _image_with_direction(R)
    with warnings_none():
        restriction_from_orientation(img, bids_phase_encoding_direction="i")


def warnings_none():
    import warnings as _w

    class _Ctx:
        def __enter__(self):
            self._cm = _w.catch_warnings()
            self._cm.__enter__()
            _w.simplefilter("error", UserWarning)
            return self

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

    return _Ctx()


def test_json_sidecar_is_read_when_direction_not_given_directly(tmp_path):
    img = _image_with_direction(IDENTITY)
    sidecar = tmp_path / "sub-01_dwi.json"
    sidecar.write_text(json.dumps({"PhaseEncodingDirection": "j-"}))
    w = restriction_from_orientation(img, json_sidecar=str(sidecar))
    assert w == (0.0, 1.0, 0.0)


def test_no_arguments_raises():
    img = _image_with_direction(IDENTITY)
    with pytest.raises(ValueError, match="Provide exactly one"):
        restriction_from_orientation(img)


def test_result_composes_directly_with_syn(tmp_path):
    """The resolver's output is a drop-in restrict_transformation for syn()."""
    import syntx

    img = _image_with_direction(IDENTITY)
    w = restriction_from_orientation(img, anatomical_axis="AP")
    base = np.zeros((16, 16, 16), dtype=np.float32)
    base[4:8, 6:12, 4:8] = 1.0
    fixed = ants.from_numpy(base)
    moving = ants.from_numpy(np.roll(base, shift=2, axis=1))
    # smoke test only: must not raise, regardless of convergence quality
    syntx.syn(fixed, moving, type_of_transform="SyNOnly", reg_iterations=[5, 5], restrict_transformation=w, verbose=False)
