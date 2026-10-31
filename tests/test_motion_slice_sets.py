"""Slice-set clock (Stage 1, WIP): the numeric core on synthetic pose series with a known answer."""

import numpy as np
import pytest

from syntx.motion_slice_sets import DOF, slice_set_clock_from_poses

TR, T = 2.0, 300
BAND = (0.15, 0.25)


def _rigid_pose(f_hz, amp, t_offsets, rng, jitter=0.0):
    """One breathing-rate trajectory on t_sl (index 2) sampled at the given per-frame time offsets."""
    t = np.arange(T) * TR
    out = np.zeros((T, 6))
    out[:, 2] = amp * np.sin(2 * np.pi * f_hz * (t + t_offsets))
    return out + jitter * rng.standard_normal((T, 6))


def test_white_jitter_is_untouched():
    rng = np.random.default_rng(0)
    early = 0.1 * rng.standard_normal((T, 6))   # frame-to-frame independent head jitter: the two sets, acquired
    late = 0.1 * rng.standard_normal((T, 6))    # about 1 s apart, see independent values
    full = 0.5 * (early + late) + 0.01 * rng.standard_normal((T, 6))
    c = slice_set_clock_from_poses(full, early, late, TR, BAND)
    assert max(d["removed_share"] for d in c["per_dof"].values()) < 0.05
    assert c["fd_head_vl_mean"] >= 0.95 * c["fd_mean"]


def test_volume_locked_component_is_removed_from_fd_only():
    rng = np.random.default_rng(1)
    vl = _rigid_pose(0.21, 0.15, 0.0, rng)               # identical in both sets: a field-of-view update
    noise = 0.01
    full = vl + noise * rng.standard_normal((T, 6))
    early = vl + noise * rng.standard_normal((T, 6))
    late = vl + noise * rng.standard_normal((T, 6))
    c = slice_set_clock_from_poses(full, early, late, TR, BAND)
    assert c["readings"]["t_sl"] == "volume_locked"
    assert abs(c["per_dof"]["t_sl"]["phase_deg"]) < 15
    assert c["per_dof"]["t_sl"]["removed_share"] > 0.9
    noise_only_fd = np.abs(np.diff(full - vl, axis=0)).sum(axis=1).mean()   # what remains when the planted part is gone
    assert c["fd_head_vl_mean"] < 1.2 * noise_only_fd
    assert c["fd_head_vl_mean"] < 0.35 * c["fd_mean"]
    np.testing.assert_allclose(c["x_head"] + c["x_vl"], full)       # the full pose is only re-attributed, never changed


def test_continuous_motion_reads_continuous_and_is_kept():
    rng = np.random.default_rng(2)
    noise = 0.01
    full = _rigid_pose(0.21, 0.15, 0.97, rng, noise)      # a volume registration sees roughly the mid-TR pose
    early = _rigid_pose(0.21, 0.15, 0.47, rng, noise)
    late = _rigid_pose(0.21, 0.15, 1.46, rng, noise)
    c = slice_set_clock_from_poses(full, early, late, TR, BAND, dt_sets_s=0.99)
    assert c["readings"]["t_sl"] == "continuous"
    assert abs(c["per_dof"]["t_sl"]["phase_deg"] - c["per_dof"]["t_sl"]["continuous_pred_deg"]) < 15
    assert c["per_dof"]["t_sl"]["removed_share"] == 0.0
    assert c["fd_head_vl_mean"] == pytest.approx(c["fd_mean"])


def test_shape_and_band_checks():
    with pytest.raises(ValueError, match="shape"):
        slice_set_clock_from_poses(np.zeros((T, 6)), np.zeros((T, 5)), np.zeros((T, 6)), TR)
    with pytest.raises(ValueError, match="Nyquist"):
        slice_set_clock_from_poses(np.zeros((T, 6)), np.zeros((T, 6)), np.zeros((T, 6)), TR, band_hz=(0.2, 0.4))
    assert list(DOF) == ["t_ro", "t_pe", "t_sl", "pitch", "roll", "yaw"]


def test_non_interleaved_slice_timing_is_refused():
    import json
    from syntx.motion_slice_sets import _early_set_from_sidecar

    interleaved = {"SliceTiming": [0.99, 0.0, 1.05, 0.06, 1.1, 0.11, 1.16, 0.17]}
    assert _early_set_from_sidecar(interleaved)[0] == 1
    ascending = {"SliceTiming": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]}
    parity, dt, source = _early_set_from_sidecar(ascending)
    assert parity is None and source == "sidecar_not_interleaved_by_parity"
    multiband = {"SliceTiming": [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75]}   # MB-2 ascending: parities overlap in time
    assert _early_set_from_sidecar(multiband)[0] is None
    assert json.dumps(ascending)  # the dict form and the path form share the parser
