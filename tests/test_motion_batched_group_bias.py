"""
tests/test_motion_batched_group_bias.py
=========================================
Tests for `syntx.motion_batched.batched_group_bias_register_pass` -- the joint per-frame
jitter + shared group-bias rigid estimator for two-acquisition-group series (e.g. a b0
series and a DWI series from the same session that carry a systematic relative offset on
top of ordinary per-frame head motion). Extracted and productionized from
`scripts/prototype_batched_joint_group_bias.py` after validation against injected
ground-truth group bias on real clinical b0/DWI data (1.238mm/1.934deg recovery error vs.
2.322mm/2.242deg for the naive two-stage "register the two mean images" alternative) --
see `docs/SESSION_2026-09-25_PHASE_CORRELATION_AND_BATCHED_MOTION_CORRECTION.md`.

These are fast, small-phantom correctness/API-contract tests; the real-data ground-truth
benchmark lives in `scripts/prototype_batched_joint_group_bias.py`.
"""

import os
import tempfile

import numpy as np
import pytest
import ants

from syntx.motion_batched import batched_group_bias_register_pass


def _shift_image(img, shift_xyz):
    """Build a translated copy of `img` via a real ants transform (matches the pull-back
    convention `batched_group_bias_register_pass` assumes: the returned fwd transform
    approximates the INVERSE of the transform used to build the frame)."""
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    tx.set_parameters(np.concatenate([np.eye(3).flatten(), -np.asarray(shift_xyz, dtype=np.float64)]))
    tx.set_fixed_parameters(np.zeros(3))
    tx_path = tempfile.mktemp(suffix=".mat")
    ants.write_transform(tx, tx_path)
    try:
        return ants.apply_transforms(fixed=img, moving=img, transformlist=[tx_path])
    finally:
        os.remove(tx_path)


def _create_group_phantom(nx=28, ny=28, nz=28):
    """Asymmetric blob (avoids rotation/reflection ambiguity) large enough for a stable
    Mattes MI landscape at this synthetic scale."""
    gx, gy, gz = np.ogrid[-nx // 2:nx // 2, -ny // 2:ny // 2, -nz // 2:nz // 2]
    base = np.exp(-((gx - 3) ** 2 / (2.0 * 5.0 ** 2) + gy ** 2 / (2.0 * 4.0 ** 2) + (gz + 2) ** 2 / (2.0 * 3.5 ** 2)))
    base += 0.5 * np.exp(-((gx + 5) ** 2 + (gy - 4) ** 2 + (gz - 3) ** 2) / (2.0 * 2.5 ** 2))
    return ants.from_numpy(base.astype(np.float32), spacing=(1.0, 1.0, 1.0))


class TestBatchedGroupBiasRegisterPass:

    def test_recovers_known_group_bias_and_frame_jitter(self):
        fixed = _create_group_phantom()
        b0_shifts = [(0.0, 0.0, 0.0), (0.6, -0.4, 0.3), (-0.5, 0.7, -0.2)]
        group_bias = (2.0, -1.5, 1.0)
        dwi_jitters = [(0.3, 0.2, -0.4), (-0.2, 0.5, 0.3), (0.4, -0.3, 0.2)]

        b0_imgs = [_shift_image(fixed, s) for s in b0_shifts]
        dwi_imgs = [_shift_image(fixed, tuple(np.array(group_bias) + np.array(j))) for j in dwi_jitters]

        all_imgs = b0_imgs + dwi_imgs
        group_mask = [False] * len(b0_imgs) + [True] * len(dwi_imgs)

        fwd, inv, R_group, t_group, elapsed = batched_group_bias_register_pass(
            fixed, all_imgs, group_mask, device="cpu", verbose=False)

        assert len(fwd) == len(all_imgs)
        assert elapsed >= 0.0

        # Recovered fwd transform is the INVERSE of the shift used to build the frame --
        # `_shift_image` applies `-shift`, so the recovered translation should be `+shift`
        # (the group-bias component included, for dwi frames, since fwd/inv here are the
        # TOTAL per-frame transforms).
        for b, true_shift in enumerate(b0_shifts):
            tx = ants.read_transform(fwd[b][0])
            t_est = np.array(tx.parameters)[9:12]
            err = np.linalg.norm(np.array(true_shift) - t_est)
            assert err < 1.0, f"b0 frame {b}: translation error {err:.3f} too large"

        for b, jitter in enumerate(dwi_jitters):
            tx = ants.read_transform(fwd[len(b0_imgs) + b][0])
            t_est = np.array(tx.parameters)[9:12]
            true_total = np.array(group_bias) + np.array(jitter)
            err = np.linalg.norm(true_total - t_est)
            assert err < 1.5, f"dwi frame {b}: total translation error {err:.3f} too large"

        # The shared group-bias parameter itself should land close to the injected value.
        group_err = np.linalg.norm(np.array(group_bias) - t_group)
        assert group_err < 1.5, f"group-bias translation error {group_err:.3f} too large"
        assert np.allclose(R_group, np.eye(3), atol=0.2)

    def test_no_group_frames_reduces_to_zero_bias(self):
        """group_mask all False: t_group/R_group must stay at their init (zero/identity),
        i.e. this degenerates cleanly to plain per-frame registration."""
        fixed = _create_group_phantom()
        shifts = [(0.0, 0.0, 0.0), (0.5, -0.3, 0.2)]
        imgs = [_shift_image(fixed, s) for s in shifts]
        fwd, inv, R_group, t_group, elapsed = batched_group_bias_register_pass(
            fixed, imgs, [False, False], device="cpu")
        assert np.allclose(t_group, 0.0, atol=1e-5)
        assert np.allclose(R_group, np.eye(3), atol=1e-5)

    def test_2d_raises_not_implemented(self):
        data = np.random.rand(16, 16).astype(np.float32)
        img2d = ants.from_numpy(data, spacing=(1.0, 1.0))
        with pytest.raises(NotImplementedError):
            batched_group_bias_register_pass(img2d, [img2d], [False])

    def test_mismatched_group_mask_length_raises(self):
        fixed = _create_group_phantom()
        with pytest.raises(ValueError, match="group_mask"):
            batched_group_bias_register_pass(fixed, [fixed, fixed], [False])

    def test_empty_input(self):
        fixed = _create_group_phantom()
        fwd, inv, R_group, t_group, elapsed = batched_group_bias_register_pass(fixed, [], [])
        assert fwd == [] and inv == [] and elapsed == 0.0
        assert np.allclose(R_group, np.eye(3)) and np.allclose(t_group, 0.0)
