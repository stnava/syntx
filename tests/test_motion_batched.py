"""
tests/test_motion_batched.py
=============================

Tests for `backend='pytorch_batched'` in `syntx.motion.motion_correction` and the
underlying `syntx.motion_batched.batched_rigid_register_pass`. See
`docs/SESSION_2026-09-25_PHASE_CORRELATION_AND_BATCHED_MOTION_CORRECTION.md` for the full
validation this backend is built on (ground-truth simulation, zero-failure large-jump
capture, ants.registration comparison). These are fast, always-run correctness/API-contract
tests; the slower real-clinical-data ground-truth benchmarks live under `scripts/`
(`bench_motion_recovery_methods.py`, `prototype_batched_motion_correction.py`).
"""

import numpy as np
import pytest
import ants

from syntx.motion import motion_correction
from syntx.motion_batched import batched_rigid_register_pass


def _create_3d_phantom(num_frames: int = 3, nx: int = 16, ny: int = 16, nz: int = 16, shifts=None):
    """Same convention as tests/test_motion.py's helper (kept independent here to avoid
    any coupling with that file's own in-progress edits)."""
    if shifts is None:
        shifts = [(0.0, 0.0, 0.0), (0.8, -0.6, 0.5), (1.5, -1.2, 1.0)]
    data = np.zeros((nx, ny, nz, num_frames), dtype=np.float32)
    gx, gy, gz = np.ogrid[-nx // 2:nx // 2, -ny // 2:ny // 2, -nz // 2:nz // 2]
    for t in range(num_frames):
        sx, sy, sz = shifts[t]
        data[..., t] = np.exp(-((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) / (2.0 * 3.5 ** 2))
    img = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0, 1.0))
    return img, shifts


class TestMotionCorrectionPytorchBatched:

    def test_recovers_known_shifts_on_small_phantom(self):
        """Regression guard for the degenerate-pyramid failure mode found and fixed in
        robust_affine.py's single-frame solver earlier this session (spurious scale/param
        drift from downsampling a small volume too far) -- motion_batched.py has its own
        MIN_VOXELS_PER_AXIS clamp; this is the case that would break without it."""
        img, shifts = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Rigid",
                                 backend="pytorch_batched", verbose=False)
        mp = res.motion_parameters
        for t in (1, 2):
            trans_err = np.linalg.norm(mp.translations[t] - np.array(shifts[t]))
            assert trans_err < 0.5, f"frame {t}: translation error {trans_err:.3f}mm too large"

    def test_matches_reference_frame_with_identity(self):
        """The reference frame itself must get an exact identity transform, same contract
        as backend='pytorch'/'ants'."""
        img, _ = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Rigid",
                                 backend="pytorch_batched", verbose=False)
        assert np.allclose(res.motion_parameters[0], 0.0, atol=1e-6)

    def test_two_pass_mean_reference(self):
        """reference='mean', two_pass=True is the default/recommended configuration and
        must work with this backend too."""
        img, shifts = _create_3d_phantom(
            num_frames=4, shifts=[(0.0, 0.0, 0.0), (0.8, -0.6, 0.5), (1.5, -1.2, 1.0), (0.5, 0.3, -0.4)])
        res = motion_correction(img, reference="mean", two_pass=True, type_of_transform="Rigid",
                                 backend="pytorch_batched", verbose=False)
        assert res.motion_corrected.shape == img.shape
        assert len(res.fwdtransforms) == 4

    def test_2d_raises_not_implemented(self):
        """2D+t is explicitly unimplemented -- must raise clearly, not silently misbehave
        or fall through to some wrong code path."""
        data = np.random.rand(16, 16, 3).astype(np.float32)
        img = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))
        with pytest.raises(NotImplementedError):
            motion_correction(img, backend="pytorch_batched", type_of_transform="Rigid")

    def test_non_rigid_transform_raises(self):
        """Only type_of_transform='Rigid' is supported so far; must fail fast with a clear
        message, not silently ignore the request."""
        img, _ = _create_3d_phantom(num_frames=3)
        with pytest.raises(ValueError, match="Rigid"):
            motion_correction(img, backend="pytorch_batched", type_of_transform="Affine")

    def test_mask_raises(self):
        """mask is not supported by any pytorch backend yet -- must fail fast."""
        img, _ = _create_3d_phantom(num_frames=3)
        mask = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))
        with pytest.raises(ValueError, match="mask"):
            motion_correction(img, backend="pytorch_batched", mask=mask)

    def test_batched_rigid_register_pass_empty_input(self):
        """Zero moving images is a degenerate but valid call (e.g. a single-frame series
        entirely covered by the identity-reference case upstream) -- must return cleanly,
        not raise."""
        fixed, _ = _create_3d_phantom(num_frames=1)
        fixed_vol = ants.slice_image(fixed, axis=3, idx=0)
        fwd, inv, elapsed = batched_rigid_register_pass(fixed_vol, [])
        assert fwd == [] and inv == [] and elapsed == 0.0

    def test_batched_rigid_register_pass_2d_raises(self):
        data = np.random.rand(16, 16).astype(np.float32)
        img2d = ants.from_numpy(data, spacing=(1.0, 1.0))
        with pytest.raises(NotImplementedError):
            batched_rigid_register_pass(img2d, [img2d])
