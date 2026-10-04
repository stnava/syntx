"""Tests for syntx.temporal_denoise -- see the module docstring for the real FDG-PET
finding (2026-10-04) this is built on: Gaussian-weighted temporal pre-averaging
stabilized both intensity-based and feature-based rigid registration on real dynamic
PET data where the raw per-frame registration was erratic.
"""

import numpy as np
import pytest
import ants

from syntx.temporal_denoise import gaussian_temporal_average, smooth_all_frames


def _noisy_series(T=10, nx=12, ny=12, nz=12, noise_std=0.3, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((nx, ny, nz), dtype=np.float32)
    gx, gy, gz = np.ogrid[-nx // 2:nx // 2, -ny // 2:ny // 2, -nz // 2:nz // 2]
    base = np.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / (2.0 * 3.0 ** 2)).astype(np.float32)
    arr = np.stack([base + rng.normal(0, noise_std, base.shape).astype(np.float32) for _ in range(T)], axis=-1)
    return ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0, 1.0)), base


class TestGaussianTemporalAverage:
    def test_reduces_noise_relative_to_raw_frame(self):
        """The whole point: averaging several noisy copies of (approximately) the same
        underlying signal should reduce residual noise relative to the ground truth."""
        img4d, base = _noisy_series(T=9, noise_std=0.4)
        center = 4
        raw_frame = img4d.numpy()[..., center]
        smoothed = gaussian_temporal_average(img4d, center=center, sigma=1.5, half_window=3)

        raw_err = np.mean((raw_frame - base) ** 2)
        smoothed_err = np.mean((smoothed.numpy() - base) ** 2)
        assert smoothed_err < raw_err

    def test_window_clipped_at_series_boundary(self):
        """center=0 with half_window=3 should still work (clipped to frames 0..3, not
        error or wrap around to the end of the series)."""
        img4d, _ = _noisy_series(T=9)
        smoothed = gaussian_temporal_average(img4d, center=0, sigma=1.5, half_window=3)
        assert smoothed.shape == img4d.shape[:3]

    def test_out_of_range_center_raises(self):
        img4d, _ = _noisy_series(T=5)
        with pytest.raises(ValueError):
            gaussian_temporal_average(img4d, center=99)

    def test_single_frame_series_is_identity(self):
        """half_window has nothing to average over -- center frame weight is 1.0."""
        img4d, base = _noisy_series(T=1)
        smoothed = gaussian_temporal_average(img4d, center=0, sigma=1.5, half_window=3)
        assert np.allclose(smoothed.numpy(), img4d.numpy()[..., 0])

    def test_none_center_delegates_to_smooth_all_frames(self):
        img4d, _ = _noisy_series(T=6)
        out_via_none = gaussian_temporal_average(img4d, center=None)
        out_via_direct = smooth_all_frames(img4d)
        assert np.allclose(out_via_none.numpy(), out_via_direct.numpy())


class TestSmoothAllFrames:
    def test_output_shape_matches_input(self):
        img4d, _ = _noisy_series(T=8)
        out = smooth_all_frames(img4d)
        assert out.shape == img4d.shape

    def test_every_frame_denoised_relative_to_ground_truth(self):
        img4d, base = _noisy_series(T=8, noise_std=0.4)
        out = smooth_all_frames(img4d)
        raw_arr = img4d.numpy()
        out_arr = out.numpy()
        for t in range(8):
            raw_err = np.mean((raw_arr[..., t] - base) ** 2)
            smoothed_err = np.mean((out_arr[..., t] - base) ** 2)
            assert smoothed_err < raw_err, f"frame {t} not improved"
