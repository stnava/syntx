"""Single-reference presets (BOLD / ASL) in ``motion_correction``. Evidence: docs/MOTION_BOLD.md."""

import ants
import numpy as np
import pytest

from syntx.motion import motion_correction
from syntx.motion_presets import get_motion_preset
from syntx.motion_reference import motion_correct_grouped


def _blob(shift=(0.0, 0.0, 0.0), n=24):
    g = np.ogrid[-n // 2:n // 2, -n // 2:n // 2, -n // 2:n // 2]
    d2 = sum((g[i] - shift[i]) ** 2 for i in range(3))
    return ants.from_numpy((100.0 * np.exp(-d2 / (2 * 3.5**2))).astype(np.float32), spacing=(1.0, 1.0, 1.0))


def _series(shifts):
    frames = [_blob(s) for s in shifts]
    return ants.list_to_ndimage(ants.from_numpy(np.zeros((24, 24, 24, len(frames)), dtype=np.float32)), frames)


def test_bold_preset_contents():
    cfg = get_motion_preset("bold")
    assert cfg == {"backend": "pytorch_batched", "seed_mode": "narrow", "schedule_mode": "medium", "sample_fraction": 0.15}


def test_bold_preset_recovers_known_shifts():
    shifts = [(0, 0, 0), (1.5, -1.0, 0.5), (-1.0, 1.0, 1.0), (0.5, 1.5, -1.0), (0.0, 0.0, 0.0)]
    res = motion_correction(_series(shifts), reference="mean", preset="bold", device="cpu")
    est = np.asarray(res["motion_parameters"].translations)
    truth = np.asarray(shifts, dtype=float)
    err = np.linalg.norm((est - est.mean(0)) - (truth - truth.mean(0)), axis=1)
    assert err.max() < 0.5
    assert res["summary"]["reference_type"] == "mean"


def test_preset_conflicts_and_wrong_entry_points_raise():
    series = _series([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    with pytest.raises(ValueError, match="do not also pass backend"):
        motion_correction(series, preset="bold", backend="ants", device="cpu")
    with pytest.raises(ValueError, match="already sets"):
        motion_correction(series, preset="bold", seed_mode="wide", device="cpu")
    with pytest.raises(ValueError, match="motion_correct_grouped"):
        motion_correction(series, preset="dwi", device="cpu")
    with pytest.raises(ValueError, match="unknown motion preset"):
        motion_correction(series, preset="nope", device="cpu")
    with pytest.raises(ValueError, match="takes the 'dwi' preset"):
        motion_correct_grouped(series, np.array([True, False, False]), preset="bold")
