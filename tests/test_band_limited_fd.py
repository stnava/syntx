"""Band-limited framewise displacement (``syntx.motion.band_limited_fd``): the near-Nyquist physiological alternation is
removed from the pose series for censoring, the full poses stay the correction. docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md S3."""
from __future__ import annotations

import numpy as np
import pytest

from syntx.motion import band_limited_fd


def _rot(T, rng, sd=0.0004):
    return np.degrees(sd * rng.standard_normal((T, 3)))


def test_known_oscillation_is_found_within_one_bin_and_removed():
    rng = np.random.default_rng(1)
    T, tr = 400, 0.8
    tt = np.arange(T) * tr
    t = 0.02 * rng.standard_normal((T, 3))
    t[:, 1] += 0.3 * np.sin(2 * np.pi * 0.3 * tt)
    r = band_limited_fd(t, _rot(T, rng), tr)
    assert r["band_source"] == "auto"
    assert abs(r["alternation_hz"] - 0.3) <= 1.0 / (T * tr) + 1e-9      # one spectral bin
    assert r["band_hz"][0] < 0.3 < r["band_hz"][1]
    assert r["fd_bandlimited_mean"] < 0.3 * np.mean(r["fd_raw_power"][1:])
    assert 0.18 <= r["alternation_amplitude_mm"] <= 0.26                 # RMS of a 0.3 mm sine is 0.21
    assert len(r["fd_bandlimited"]) == T and r["fd_bandlimited"][0] == 0.0


def test_white_jitter_and_real_motion_are_left_alone():
    rng = np.random.default_rng(2)
    T, tr = 400, 0.8
    white = 0.05 * rng.standard_normal((T, 3))
    r = band_limited_fd(white, _rot(T, rng), tr)
    assert r["band_source"] == "none" and r["band_hz"] is None
    assert np.allclose(r["fd_bandlimited"], r["fd_raw_power"])
    t = np.zeros((T, 3))
    t[:, 0] = np.linspace(0, 1, T)            # slow drift
    t[200:, 2] += 0.8                         # one real jump
    t += 0.01 * rng.standard_normal((T, 3))
    r = band_limited_fd(t, _rot(T, rng), tr)
    assert r["band_source"] == "none"
    assert np.allclose(r["fd_bandlimited"], r["fd_raw_power"])
    assert r["fd_bandlimited"][200] > 0.7     # the jump is kept


def test_given_band_is_applied_and_recorded():
    rng = np.random.default_rng(3)
    T, tr = 300, 1.0
    t = 0.05 * rng.standard_normal((T, 3))
    r = band_limited_fd(t, _rot(T, rng), tr, band_hz=(0.2, 0.4))
    assert r["band_source"] == "given" and r["band_hz"] == (0.2, 0.4)
    assert r["fd_bandlimited_mean"] < np.mean(r["fd_raw_power"][1:])


def test_short_series_and_bad_tr():
    rng = np.random.default_rng(4)
    r = band_limited_fd(0.1 * rng.standard_normal((5, 3)), _rot(5, rng), 2.0)
    assert r["band_source"] == "none" and len(r["fd_bandlimited"]) == 5
    with pytest.raises(ValueError):
        band_limited_fd(np.zeros((20, 3)), np.zeros((20, 3)), 0.0)


def test_keys_attach_through_motion_correction_and_the_design_engines():
    """The 4-D time spacing is the TR; every engine's result carries ``fd_bandlimited`` and ``fd_band``."""
    import syntx
    from syntx.motion_simulation import simulate_asl_series

    sim = simulate_asl_series(n_frames=12, seed=0)
    img = sim.image
    assert img.dimension == 4 and img.spacing[3] > 0
    mc = syntx.motion_correction(img, reference="mean", type_of_transform="Rigid", preset="asl", verbose=False)
    assert mc["fd_band"]["tr_s"] == pytest.approx(float(img.spacing[3]))
    assert mc["fd_band"]["band_source"] in ("auto", "none")
    assert len(mc["fd_bandlimited"]) == 12 and "fd_bandlimited_mean" in mc["summary"]
    vt = ["control" if i % 2 == 0 else "label" for i in range(12)]
    design = syntx.make_design("tag_control", volume_types=vt)
    for engine in ("unified", "joint"):
        r = syntx.motion_correct_design(img, design, engine=engine, device="cpu", verbose=False)
        assert len(r["fd_bandlimited"]) == 12 and r["fd_band"]["band_source"] in ("auto", "none"), engine
