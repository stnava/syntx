"""Tests for syntx.motion_reference -- low-motion-subset reference building and
contrast-grouped motion correction (e.g. DWI b0-then-DWI). See the module's own
docstring for the real-data findings (2026-10-04) this design is built on.
"""

import numpy as np
import pytest
import ants

from syntx.motion_reference import build_low_motion_reference, motion_correct_grouped


def _phantom_frame(nx=20, ny=20, nz=20, shift=(0.0, 0.0, 0.0)):
    gx, gy, gz = np.ogrid[-nx // 2:nx // 2, -ny // 2:ny // 2, -nz // 2:nz // 2]
    sx, sy, sz = shift
    data = np.exp(-((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) / (2.0 * 3.5 ** 2)).astype(np.float32)
    return ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))


class TestBuildLowMotionReference:
    def test_single_frame_returns_clone(self):
        im = _phantom_frame()
        ref = build_low_motion_reference([im])
        assert ref.shape == im.shape
        assert np.allclose(ref.numpy(), im.numpy())

    def test_fewer_frames_than_n_low_motion_uses_all(self):
        imgs = [_phantom_frame(shift=s) for s in [(0, 0, 0), (0.3, 0.1, -0.2), (0.5, -0.4, 0.3)]]
        ref = build_low_motion_reference(imgs, n_low_motion=10, verbose=False)
        assert ref.shape == imgs[0].shape
        # sharpened reference should correlate well with any single well-aligned frame
        corr = np.corrcoef(ref.numpy().ravel(), imgs[0].numpy().ravel())[0, 1]
        assert corr > 0.8

    def test_subset_selection_prefers_low_motion_frames(self):
        """With a large series dominated by low-motion frames and a few big outliers,
        the built reference should resemble the low-motion frames much more than the
        outliers (i.e. selection is actually doing something, not just averaging
        everything blindly)."""
        low_motion = [_phantom_frame(shift=(0.0, 0.0, 0.0)) for _ in range(8)]
        outliers = [_phantom_frame(shift=(6.0, 5.0, -6.0)) for _ in range(4)]
        imgs = low_motion + outliers
        ref = build_low_motion_reference(imgs, n_low_motion=8, verbose=False)

        corr_low = np.corrcoef(ref.numpy().ravel(), low_motion[0].numpy().ravel())[0, 1]
        corr_outlier = np.corrcoef(ref.numpy().ravel(), outliers[0].numpy().ravel())[0, 1]
        assert corr_low > corr_outlier

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_low_motion_reference([])


class TestMotionCorrectGrouped:
    """Two-mean design: build a sharp mean per group, align the two means to each other
    ONCE (the only cross-contrast registration anywhere in the pipeline), then motion-
    correct each group's own frames to its OWN mean (same-contrast, small-motion).
    Replaced an earlier, wrong single-shared-reference design after real SOCOM DWI data
    showed it made b0 apparent motion WORSE (7.28mm), not better, than the original
    blended-mean method (4.66mm) -- the two-mean design instead matches a trusted
    b0-only-internal baseline almost exactly (0.796mm vs 0.748mm, real data, see
    docs/ADAPTIVE_MOTION_CORRECTION_AND_RECOVERY.html)."""

    def test_basic_contract(self):
        frames = [_phantom_frame(shift=s) for s in
                  [(0, 0, 0), (0.1, 0, 0), (2.0, -1.5, 1.0), (1.8, -1.2, 0.9), (2.2, -1.6, 1.1)]]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 5), dtype=np.float32)), frames)
        group_is_reference = np.array([True, True, False, False, False])

        out = motion_correct_grouped(img4d, group_is_reference, n_low_motion=10, verbose=False)
        assert out.motion_corrected.shape == img4d.shape
        assert len(out.fd) == 5
        # The two-mean-specific artifacts should all be present (not just the generic
        # motion_corrected/fd contract) -- a caller may want the per-group detail.
        for key in ("group_a_result", "group_b_result", "group_a_mean", "group_b_mean", "cross_registration"):
            assert key in out

    def test_mismatched_group_length_raises(self):
        img4d = ants.from_numpy(np.zeros((10, 10, 10, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            motion_correct_grouped(img4d, np.array([True, False, True]))

    def test_no_group_a_frames_raises(self):
        img4d = ants.from_numpy(np.zeros((10, 10, 10, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            motion_correct_grouped(img4d, np.array([False, False, False, False]))

    def test_no_group_b_frames_raises(self):
        """New validation (the two-mean design needs BOTH groups non-empty -- there must
        be a group B mean to align to group A in the one cross-registration step)."""
        img4d = ants.from_numpy(np.zeros((10, 10, 10, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            motion_correct_grouped(img4d, np.array([True, True, True, True]))
