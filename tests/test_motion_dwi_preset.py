"""The DWI motion preset: b0_mean and dwi_mean, per-group registration, one cross-contrast registration, single resampling,
series-level Jenkinson FD. Real-data evidence: docs/MOTION_DWI.md."""

import ants
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from syntx.motion import MotionParameters, calculate_framewise_displacement, jenkinson_framewise_displacement
from syntx.motion_presets import MOTION_PRESETS, get_motion_preset
from syntx.motion_reference import motion_correct_grouped, shell_groups


def _blob(shift=(0.0, 0.0, 0.0), n=20, scale=1.0):
    g = np.ogrid[-n // 2:n // 2, -n // 2:n // 2, -n // 2:n // 2]
    d2 = sum((g[i] - shift[i]) ** 2 for i in range(3))
    return ants.from_numpy((scale * np.exp(-d2 / (2.0 * 3.5**2))).astype(np.float32), spacing=(1.0, 1.0, 1.0))


def _homog(rot_deg, trans, centre=np.zeros(3)):
    r = Rot.from_euler("xyz", np.radians(rot_deg)).as_matrix()
    h = np.eye(4)
    h[:3, :3] = r
    h[:3, 3] = trans + centre - r @ centre
    return h


class TestJenkinsonFd:
    def test_identity_is_zero_and_translation_is_exact(self):
        assert np.allclose(jenkinson_framewise_displacement([np.eye(4)] * 5), 0.0)
        fd = jenkinson_framewise_displacement([_homog([0, 0, 0], [0, 0, 0]), _homog([0, 0, 0], [2.0, 0, 0])])
        assert fd[0] == 0.0 and fd[1] == pytest.approx(2.0)

    def test_matches_monte_carlo_ball(self):
        rng = np.random.default_rng(2)
        c = np.array([4.0, 3.7, 22.7])
        hs = [_homog(rng.normal(0, 1, 3), rng.normal(0, 0.5, 3), c) for _ in range(6)]
        fd = jenkinson_framewise_displacement(hs, 50.0, c)
        pts = rng.normal(size=(300000, 3))
        pts = c + 50.0 * pts / np.linalg.norm(pts, axis=1, keepdims=True) * rng.random((300000, 1)) ** (1 / 3)
        for k in range(1, 6):
            m = hs[k] @ np.linalg.inv(hs[k - 1])
            ref = np.sqrt((((pts @ m[:3, :3].T + m[:3, 3]) - pts) ** 2).sum(1).mean())
            assert fd[k] == pytest.approx(ref, rel=0.01)

    def test_power_fd_unchanged_and_jenkinson_shared(self):
        rng = np.random.default_rng(3)
        t, r = rng.normal(0, 0.5, (6, 3)), np.radians(rng.normal(0, 1, (6, 3)))
        hs = [_homog(np.degrees(r[k]), t[k]) for k in range(6)]
        mp = MotionParameters(np.hstack([t, r]), columns=["tx", "ty", "tz", "rx", "ry", "rz"],
                              rad_to_deg_cols=["rx", "ry", "rz"], spatial_dim=3)
        fp, fj = calculate_framewise_displacement(mp, hs, radius=50.0, center_of_sphere=np.zeros(3))
        by_hand = [abs(t[k] - t[k - 1]).sum() + 50.0 * abs(r[k] - r[k - 1]).sum() for k in range(1, 6)]
        assert np.allclose(fp[1:], by_hand)
        assert np.allclose(fj, jenkinson_framewise_displacement(hs, 50.0, np.zeros(3)))


class TestPresetPlumbing:
    def test_registry_and_unknown_preset(self):
        assert set(MOTION_PRESETS) == {"dwi", "bold"}
        assert get_motion_preset("dwi") == {"backend": "pytorch_batched", "resolution_cap_mm": 2.0, "interpolator": "bSpline", "cross_num_bins": 128, "seed_mode": "narrow", "schedule_mode": "full"}
        with pytest.raises(ValueError, match="unknown motion preset"):
            get_motion_preset("nope")


def test_dwi_preset_recovers_known_shifts_on_direction_dependent_contrast():
    """Small synthetic DWI: 2 b0 + 12 diffusion volumes with direction-dependent contrast, displaced by known sub-voxel
    shifts; the preset must recover the relative shifts and return the series FD in acquisition order."""
    rng = np.random.default_rng(5)
    nd = 12
    d = rng.normal(size=(nd, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    shifts = rng.normal(0, 0.8, (nd, 3))
    scale = 0.5 + 0.5 * (d @ np.array([0.0, 1.0, 0.0])) ** 2
    frames = [_blob()] * 2 + [_blob(shift=tuple(s), scale=sc) for s, sc in zip(shifts, scale)]
    img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, nd + 2), dtype=np.float32)), frames)
    b = np.array([0, 0] + [1000] * nd, dtype=float)
    out = motion_correct_grouped(img4d, b <= 50, preset="dwi", device="cpu")
    assert out["preset"] == "dwi"
    assert len(out["fd"]) == nd + 2 and out["fd"][0] == 0.0 and np.all(np.isfinite(out["fd"]))
    est = np.asarray(out["group_results"][1].motion_parameters.translations)
    err = np.linalg.norm((est - est.mean(0)) - (shifts - shifts.mean(0)), axis=1)
    assert err.max() < 0.5


def test_dwi_preset_cross_registration_uses_the_corrected_group_means():
    """The b0 <-> diffusion registration is computed once, between the means of the corrected frames of each group."""
    frames = [_blob()] * 3 + [_blob(shift=(1.0, 0.0, 0.0), scale=0.6)] * 4
    img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 7), dtype=np.float32)), frames)
    out = motion_correct_grouped(img4d, np.array([True] * 3 + [False] * 4), preset="dwi", device="cpu")
    tx = ants.read_transform(out["cross_registrations"][0]["fwdtransforms"][0])
    assert abs(abs(np.asarray(tx.parameters)[9]) - 1.0) < 0.3  # the 1 voxel shift between the two groups is recovered


def test_shell_groups_labels_b0_zero_and_shells_ascending():
    b = np.array([0, 1000, 2950, 3000, 1005, 2000, 0, 995, 2010, 3000])
    assert shell_groups(b).tolist() == [0, 1, 3, 3, 1, 2, 0, 1, 2, 3]
    assert shell_groups(np.array([0, 5, 1000, 1000])).tolist() == [0, 0, 1, 1]  # b <= 50 counts as b0


def test_three_groups_chain_their_cross_registrations_and_return_full_chains():
    """b0, a 'b1000'-like and a 'b3000'-like group (different contrast, 1 and 2 voxel shifts): the chain composes."""
    frames = [_blob()] * 3 + [_blob(shift=(1.0, 0, 0), scale=0.6)] * 3 + [_blob(shift=(2.0, 0, 0), scale=0.3)] * 3
    img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 9), dtype=np.float32)), frames)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    out = motion_correct_grouped(img4d, labels, preset="dwi", device="cpu")
    assert len(out["cross_registrations"]) == 2 and len(out["group_results"]) == 3
    assert [len(c) for c in out["fwdtransforms"]] == [1, 1, 1, 2, 2, 2, 3, 3, 3]  # [own] / [cross_1, own] / [cross_1, cross_2, own]
    assert out["motion_corrected"].shape == img4d.shape and len(out["fd"]) == 9


def test_grouped_validates_its_labels():
    img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 4), dtype=np.float32)), [_blob()] * 4)
    with pytest.raises(ValueError, match="labels 0..k-1"):
        motion_correct_grouped(img4d, np.array([0, 0, 2, 2]), device="cpu")
    with pytest.raises(ValueError, match="one entry per frame"):
        motion_correct_grouped(img4d, np.array([0, 1]), device="cpu")
