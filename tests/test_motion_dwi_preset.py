"""Tests for the DWI motion preset: leave-one-out direction targets, per-frame references, series-level
Jenkinson FD, and the preset plumbing in ``motion_correct_grouped``. Real-data evidence: docs/MOTION_DWI.md."""

import ants
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from syntx.motion import calculate_framewise_displacement, jenkinson_framewise_displacement, MotionParameters, motion_correction
from syntx.motion_presets import MOTION_PRESETS, get_motion_preset
from syntx.motion_reference import dwi_direction_targets, motion_correct_grouped


def _blob(shift=(0.0, 0.0, 0.0), n=20, scale=1.0):
    g = np.ogrid[-n // 2:n // 2, -n // 2:n // 2, -n // 2:n // 2]
    d2 = sum((g[i] - shift[i]) ** 2 for i in range(3))
    return ants.from_numpy((scale * np.exp(-d2 / (2.0 * 3.5**2))).astype(np.float32), spacing=(1.0, 1.0, 1.0))


def _dirs(n, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class TestDwiDirectionTargets:
    def test_target_never_contains_its_own_volume(self):
        d = _dirs(8)
        frames = [_blob(scale=0.5 + 0.1 * i) for i in range(8)]
        t0 = dwi_direction_targets(frames, np.full(8, 1000.0), d)
        frames2 = list(frames)
        frames2[3] = _blob(scale=50.0)  # change ONLY volume 3
        t1 = dwi_direction_targets(frames2, np.full(8, 1000.0), d)
        assert np.allclose(t0[3].numpy(), t1[3].numpy())  # volume 3's own target is unchanged
        assert not np.allclose(t0[4].numpy(), t1[4].numpy())  # the others do see it

    def test_recovers_a_smooth_direction_dependent_signal(self):
        d = _dirs(30, seed=1)
        axis = np.array([1.0, 0.0, 0.0])
        scale = 0.4 + 0.6 * (d @ axis) ** 2
        frames = [_blob(scale=s) for s in scale]
        targets = dwi_direction_targets(frames, np.full(30, 1000.0), d)
        err = np.array([abs(t.numpy().max() - f.numpy().max()) for t, f in zip(targets, frames)])
        mean_err = abs(np.mean([f.numpy().max() for f in frames]) - np.array([f.numpy().max() for f in frames])).mean()
        assert err.mean() < 0.5 * mean_err  # better than predicting the shell mean

    def test_matches_the_direct_per_volume_formula(self):
        """The matrix form equals the plain definition: GP regression of the other volumes of the shell onto the
        direction of volume i, with the shell mean of the others as prior mean."""
        rng = np.random.default_rng(3)
        n = 9
        d = _dirs(n, seed=6)
        b = np.array([1000.0] * 5 + [2000.0] * 4)
        frames = [ants.from_numpy(rng.random((6, 7, 8)).astype(np.float32)) for _ in range(n)]
        targets = dwi_direction_targets(frames, b, d, kappa=6.0, ridge=0.05)
        stack = np.stack([f.numpy() for f in frames], axis=-1)
        for i in range(n):
            idx = [j for j in range(n) if b[j] == b[i] and j != i]
            kmat = np.exp(6.0 * ((d[idx] @ d[idx].T) ** 2 - 1.0)) + 0.05 * np.eye(len(idx))
            w = np.linalg.solve(kmat, np.exp(6.0 * ((d[idx] @ d[i]) ** 2 - 1.0)))
            mu = stack[..., idx].mean(-1)
            expected = mu + np.tensordot(stack[..., idx] - mu[..., None], w, axes=([3], [0]))
            assert np.allclose(targets[i].numpy(), expected, atol=1e-5)

    def test_shells_are_separate(self):
        d = _dirs(8)
        b = np.array([1000.0] * 4 + [3000.0] * 4)
        frames = [_blob(scale=1.0)] * 4 + [_blob(scale=0.2)] * 4
        t = dwi_direction_targets(frames, b, d)
        assert np.allclose(t[0].numpy().max(), 1.0, atol=1e-3) and np.allclose(t[7].numpy().max(), 0.2, atol=1e-3)

    def test_errors(self):
        d = _dirs(4)
        with pytest.raises(ValueError, match=">= 2"):
            dwi_direction_targets([_blob()] * 3, np.array([1000.0, 3000.0, 3000.0]), d[:3])
        with pytest.raises(ValueError, match="zero gradient"):
            dwi_direction_targets([_blob()] * 3, np.full(3, 1000.0), np.zeros((3, 3)))
        with pytest.raises(ValueError, match="one bval"):
            dwi_direction_targets([_blob()] * 3, np.full(2, 1000.0), d[:3])


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
        assert get_motion_preset("dwi")["backend"] == "pytorch_batched"
        with pytest.raises(ValueError, match="unknown motion preset"):
            get_motion_preset("nope")

    def test_preset_needs_gradients(self):
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 3), dtype=np.float32)), [_blob()] * 3)
        with pytest.raises(ValueError, match="needs bvals and bvecs"):
            motion_correct_grouped(img4d, np.array([True, False, False]), preset="dwi")

    def test_per_frame_reference_validation(self):
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 3), dtype=np.float32)), [_blob()] * 3)
        with pytest.raises(ValueError, match="one per frame"):
            motion_correction(img4d, reference=[_blob()] * 2, backend="pytorch_batched", device="cpu")
        with pytest.raises(ValueError, match="pytorch_batched"):
            motion_correction(img4d, reference=[_blob()] * 3, backend="pytorch_batched_adaptive", device="cpu")


def test_dwi_preset_recovers_known_shifts_on_direction_dependent_contrast():
    """Small synthetic DWI: 2 b0 + 12 diffusion volumes whose contrast depends on direction and which are
    displaced by known sub-voxel shifts; the preset must recover the relative shifts and report series FD."""
    rng = np.random.default_rng(5)
    nd = 12
    d = _dirs(nd, seed=4)
    shifts = rng.normal(0, 0.8, (nd, 3))
    scale = 0.5 + 0.5 * (d @ np.array([0.0, 1.0, 0.0])) ** 2
    frames = [_blob()] * 2 + [_blob(shift=tuple(s), scale=sc) for s, sc in zip(shifts, scale)]
    img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, nd + 2), dtype=np.float32)), frames)
    b = np.array([0, 0] + [1000] * nd, dtype=float)
    bvecs = np.vstack([np.zeros((2, 3)), d])
    out = motion_correct_grouped(img4d, b <= 50, preset="dwi", bvals=b, bvecs=bvecs, device="cpu")
    assert out["preset"] == "dwi"
    assert len(out["fd"]) == nd + 2 and out["fd"][0] == 0.0 and np.all(np.isfinite(out["fd"]))
    est = np.asarray(out["group_b_result"].motion_parameters.translations)
    # the fixed->moving transform translates by the injected shift (up to one common offset): compare after
    # removing means
    err = np.linalg.norm((est - est.mean(0)) - (shifts - shifts.mean(0)), axis=1)
    assert err.max() < 0.5
