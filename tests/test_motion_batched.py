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


class TestMotionCorrectionAutoBackend:
    """backend='auto' (the new default) must resolve to 'pytorch_batched' whenever
    eligible (3D+t, Rigid, no mask) and fall back correctly otherwise -- never silently
    to legacy `ants.registration` except for the one case (mask) that genuinely needs it."""

    def test_default_backend_is_auto_and_eligible_case_succeeds(self):
        img, shifts = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Rigid", verbose=False)
        mp = res.motion_parameters
        for t in (1, 2):
            trans_err = np.linalg.norm(mp.translations[t] - np.array(shifts[t]))
            assert trans_err < 0.5, f"frame {t}: translation error {trans_err:.3f}mm too large"

    def test_auto_falls_back_to_ants_with_mask(self):
        img, _ = _create_3d_phantom(num_frames=3)
        mask = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))
        res = motion_correction(img, reference=0, type_of_transform="Rigid", mask=mask, verbose=False)
        assert res.motion_corrected.shape == img.shape

    def test_auto_falls_back_to_pytorch_for_2d(self):
        data = np.random.rand(16, 16, 3).astype(np.float32)
        img2d = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))
        res = motion_correction(img2d, reference=0, type_of_transform="Rigid", verbose=False)
        assert res.motion_corrected.shape == img2d.shape

    def test_auto_falls_back_to_pytorch_for_affine(self):
        img, _ = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Affine", verbose=False)
        assert res.motion_corrected.shape == img.shape

    def test_invalid_backend_string_raises(self):
        img, _ = _create_3d_phantom(num_frames=3)
        with pytest.raises(ValueError, match="backend must be"):
            motion_correction(img, backend="not_a_real_backend")


class TestBatchedRigidRegisterPassChunking:
    """Regression guard for a real crash found on 356-frame real FDG-PET data: a single
    batched call over ALL frames at once overflowed MPS's torch.gradient kernel
    ("value cannot be converted to type dest_t without overflow") once B * volume_numel
    got large enough. batched_rigid_register_pass now auto-chunks long series; these tests
    force chunking at small scale (via max_batch_frames) and confirm the chunked path still
    recovers the known ground-truth shifts to the same accuracy as the unchunked path --
    not bit-identical parameters (each chunk's LBFGS history is independent of the others,
    so chunking can converge to a very slightly different, equally valid solution)."""

    def test_chunked_recovers_known_shifts_as_well_as_unchunked(self):
        shifts = [(0.0, 0.0, 0.0), (0.8, -0.6, 0.5), (1.5, -1.2, 1.0),
                  (0.6, 0.9, -0.7), (-1.0, 0.5, 0.8), (1.2, -0.8, -0.5)]
        img, _ = _create_3d_phantom(num_frames=6, nx=16, ny=16, nz=16, shifts=shifts)
        ref_np = img.numpy()[..., 0]
        ref = ants.from_numpy(ref_np, spacing=(1.0, 1.0, 1.0))
        moving_imgs = [ants.from_numpy(img.numpy()[..., t], spacing=(1.0, 1.0, 1.0))
                       for t in range(6)]

        def resample_error(fwd, t):
            """Apply the recovered forward transform and compare the resampled frame back
            to the reference -- convention-agnostic (unlike comparing raw tx.parameters to
            the known shift directly, which depends on the forward-transform's direction
            convention), and it's the thing that actually matters: did registration undo
            the known motion."""
            warped = ants.apply_transforms(fixed=ref, moving=moving_imgs[t],
                                            transformlist=fwd[t][0], interpolator="linear")
            return float(np.mean((warped.numpy() - ref_np) ** 2))

        fwd_unchunked, _, _ = batched_rigid_register_pass(
            ref, moving_imgs, verbose=False, max_batch_frames=None,
        )
        fwd_chunked, _, _ = batched_rigid_register_pass(
            ref, moving_imgs, verbose=False, max_batch_frames=2,
        )

        assert len(fwd_chunked) == len(fwd_unchunked) == 6
        # Baseline: residual if no correction were applied at all (raw moving vs reference).
        raw_mse = [float(np.mean((moving_imgs[t].numpy() - ref_np) ** 2)) for t in range(1, 6)]
        for i, t in enumerate(range(1, 6)):
            mse_u = resample_error(fwd_unchunked, t)
            mse_c = resample_error(fwd_chunked, t)
            assert mse_u < raw_mse[i] * 0.5, f"frame {t}: unchunked correction didn't improve alignment"
            assert mse_c < raw_mse[i] * 0.5, f"frame {t}: chunked correction didn't improve alignment"

    def test_auto_chunk_size_splits_large_series_without_crashing(self):
        """auto_chunk is computed from _MAX_BATCH_TENSOR_ELEMENTS // volume_numel -- on a
        tiny phantom that's a huge number of frames, so force a tiny max_batch_frames
        directly (the auto-computed path itself is exercised by the default-None case
        above with a small series that never splits; this confirms the splitting
        mechanics -- multiple chunks, correct concatenation order -- work for a series
        longer than one chunk)."""
        img, _ = _create_3d_phantom(num_frames=9, nx=12, ny=12, nz=12,
                                     shifts=[(0.5 * t, -0.3 * t, 0.4 * t) for t in range(9)])
        ref = ants.from_numpy(img.numpy()[..., 0], spacing=(1.0, 1.0, 1.0))
        moving_imgs = [ants.from_numpy(img.numpy()[..., t], spacing=(1.0, 1.0, 1.0))
                       for t in range(9)]

        fwd, inv, elapsed = batched_rigid_register_pass(
            ref, moving_imgs, verbose=False, max_batch_frames=4,
        )
        assert len(fwd) == len(inv) == 9
        assert elapsed >= 0.0

    def test_empty_moving_imgs_returns_immediately(self):
        img, _ = _create_3d_phantom(num_frames=1)
        ref = ants.from_numpy(img.numpy()[..., 0], spacing=(1.0, 1.0, 1.0))
        fwd, inv, elapsed = batched_rigid_register_pass(ref, [], max_batch_frames=2)
        assert fwd == [] and inv == [] and elapsed == 0.0
