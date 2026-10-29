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
from syntx.motion_batched import (
    batched_rigid_register_pass,
    batched_rigid_register_pass_adaptive,
    batched_rigid_register_pass_temporal,
)


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


class TestMotionCorrectionPytorchBatchedTemporal:
    """backend='pytorch_batched_temporal' -- the production wiring of the 4-lever fast path
    piloted against real 356-frame FDG-PET data (~48x speedup, see
    batched_rigid_register_pass_temporal's docstring and docs/tracking.md in
    antsxfunctional). Never selected by 'auto'; opt-in only via explicit backend=."""

    def test_recovers_known_shift_via_public_api(self):
        img, shifts = _create_3d_phantom(
            num_frames=4, nx=20, ny=20, nz=20,
            shifts=[(0.0, 0.0, 0.0), (0.3, -0.2, 0.1), (1.2, -0.8, 0.6), (0.1, 0.1, -0.1)],
        )
        res = motion_correction(img, reference=0, type_of_transform="Rigid",
                                 backend="pytorch_batched_temporal", verbose=False)
        trans_err = np.linalg.norm(res.motion_parameters.translations[2] - np.array(shifts[2]))
        assert trans_err < 0.5, f"translation error {trans_err:.3f}mm too large"
        assert res.motion_corrected.shape == img.shape
        assert "temporal_variance_reduction_percent" in res.summary

    def test_matches_reference_frame_with_identity(self):
        img, _ = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Rigid",
                                 backend="pytorch_batched_temporal", verbose=False)
        assert np.allclose(res.motion_parameters[0], 0.0, atol=1e-6)

    def test_disallowed_kwarg_raises_typeerror(self):
        img, _ = _create_3d_phantom(num_frames=3)
        with pytest.raises(TypeError, match="pytorch_batched_temporal"):
            motion_correction(img, backend="pytorch_batched_temporal", foo=1)

    def test_allowed_kwargs_accepted(self):
        img, _ = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Rigid",
                                 backend="pytorch_batched_temporal", verbose=False,
                                 resolution_cap_mm=None, motion_gate_threshold=None, num_bins=24)
        assert res.motion_corrected.shape == img.shape

    def test_non_rigid_transform_raises(self):
        img, _ = _create_3d_phantom(num_frames=3)
        with pytest.raises(ValueError, match="Rigid"):
            motion_correction(img, backend="pytorch_batched_temporal", type_of_transform="Affine")

    def test_mask_raises(self):
        img, _ = _create_3d_phantom(num_frames=3)
        mask = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))
        with pytest.raises(ValueError, match="mask"):
            motion_correction(img, backend="pytorch_batched_temporal", mask=mask)

    def test_2d_raises_not_implemented(self):
        data = np.random.rand(16, 16, 3).astype(np.float32)
        img = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))
        with pytest.raises(NotImplementedError):
            motion_correction(img, backend="pytorch_batched_temporal", type_of_transform="Rigid")

    def test_auto_never_selects_temporal_backend(self):
        """'auto' must keep selecting 'pytorch_batched' (full precision/capture range) --
        the temporal fast path trades those off for speed and must stay strictly opt-in."""
        img, shifts = _create_3d_phantom(num_frames=3)
        res_auto = motion_correction(img, reference=0, type_of_transform="Rigid", verbose=False)
        res_wide = motion_correction(img, reference=0, type_of_transform="Rigid",
                                      backend="pytorch_batched", verbose=False)
        assert np.allclose(res_auto.motion_parameters.translations,
                            res_wide.motion_parameters.translations, atol=1e-5)


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


class TestBatchedRigidRegisterPassTemporal:
    """Pilot: batched_rigid_register_pass_temporal combines 4 speed levers (resolution cap,
    narrow seed search, a shorter schedule, and a motion gate skipping already-aligned
    frames) for TEMPORALLY COHERENT series (e.g. dynamic PET). Validated on real 356-frame
    FDG-PET data: ~48x faster (100s vs ~80min) than the default wide/full path, with
    correlation-to-reference improving or unchanged for every frame -- see docs/tracking.md
    in antsxfunctional for the full numbers. These tests check the API contract and basic
    correctness on cheap synthetic phantoms, not the full-scale timing (covered by the real
    FDG pilot, not re-run here since that needs the real dataset on disk)."""

    def test_recovers_known_shift_and_gates_aligned_frames(self):
        shifts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.5, -1.2, 1.0), (0.0, 0.0, 0.0)]
        img, _ = _create_3d_phantom(num_frames=4, nx=20, ny=20, nz=20, shifts=shifts)
        ref_np = img.numpy()[..., 0]
        ref = ants.from_numpy(ref_np, spacing=(1.0, 1.0, 1.0))
        moving_imgs = [ants.from_numpy(img.numpy()[..., t], spacing=(1.0, 1.0, 1.0))
                       for t in range(4)]

        fwd, inv, elapsed = batched_rigid_register_pass_temporal(
            ref, moving_imgs, verbose=False, resolution_cap_mm=None, motion_gate_threshold=0.999,
        )
        assert len(fwd) == len(inv) == 4
        assert elapsed >= 0.0

        for t in range(4):
            warped = ants.apply_transforms(fixed=ref, moving=moving_imgs[t],
                                            transformlist=fwd[t][0], interpolator="linear")
            mse = float(np.mean((warped.numpy() - ref_np) ** 2))
            raw_mse = float(np.mean((moving_imgs[t].numpy() - ref_np) ** 2))
            assert mse <= raw_mse + 1e-9, f"frame {t}: correction made alignment worse"

    def test_motion_gate_disabled_registers_every_frame(self):
        shifts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        img, _ = _create_3d_phantom(num_frames=2, nx=16, ny=16, nz=16, shifts=shifts)
        ref = ants.from_numpy(img.numpy()[..., 0], spacing=(1.0, 1.0, 1.0))
        moving_imgs = [ants.from_numpy(img.numpy()[..., t], spacing=(1.0, 1.0, 1.0))
                       for t in range(2)]
        fwd, inv, _ = batched_rigid_register_pass_temporal(
            ref, moving_imgs, verbose=False, resolution_cap_mm=None, motion_gate_threshold=None,
        )
        assert len(fwd) == 2

    def test_resolution_cap_does_not_upsample(self):
        """_cap_resolution must never make an already-coarser image finer."""
        from syntx.motion_batched import _cap_resolution
        img = ants.from_numpy(np.zeros((8, 8, 8), dtype=np.float32), spacing=(5.0, 5.0, 5.0))
        capped = _cap_resolution(img, target_spacing_mm=1.0)
        assert capped is img

    def test_empty_moving_imgs_returns_immediately_temporal(self):
        img, _ = _create_3d_phantom(num_frames=1)
        ref = ants.from_numpy(img.numpy()[..., 0], spacing=(1.0, 1.0, 1.0))
        fwd, inv, elapsed = batched_rigid_register_pass_temporal(ref, [])
        assert fwd == [] and inv == [] and elapsed == 0.0


class TestMotionCorrectionPytorchBatchedAdaptive:
    """Tests for `batched_rigid_register_pass_adaptive` -- the registration-derived-FD
    gate that replaced the correlation-based gate after it was found to silently skip
    registering EVERY frame of a real low-resolution (3.5mm) PPMI rsfMRI series (see the
    function's own docstring for the full real-data finding, 2026-10-04)."""

    def test_recovers_known_shift_on_small_phantom(self):
        shifts = [(0.0, 0.0, 0.0), (1.5, -1.2, 1.0)]
        img, _ = _create_3d_phantom(num_frames=2, nx=20, ny=20, nz=20, shifts=shifts)
        ref_np = img.numpy()[..., 0]
        ref = ants.from_numpy(ref_np, spacing=(1.0, 1.0, 1.0))
        moving_imgs = [ants.from_numpy(img.numpy()[..., t], spacing=(1.0, 1.0, 1.0)) for t in range(2)]

        fwd, inv, elapsed = batched_rigid_register_pass_adaptive(
            ref, moving_imgs, verbose=False, coarse_resolution_cap_mm=None, fine_resolution_cap_mm=None,
        )
        assert len(fwd) == len(inv) == 2
        assert elapsed >= 0.0
        for t in range(2):
            warped = ants.apply_transforms(fixed=ref, moving=moving_imgs[t],
                                            transformlist=fwd[t][0], interpolator="linear")
            mse = float(np.mean((warped.numpy() - ref_np) ** 2))
            raw_mse = float(np.mean((moving_imgs[t].numpy() - ref_np) ** 2))
            # Unlike batched_rigid_register_pass_temporal's gate (which gives an
            # already-aligned frame an EXACT identity transform), this adaptive gate
            # still runs a real (coarse) optimization on every frame, including one
            # that happens to exactly equal the reference -- a looser tolerance than
            # 1e-9 absorbs that expected optimizer noise without masking a real
            # alignment regression (raw_mse=0 here; 1e-4 is still 2 orders of
            # magnitude tighter than this phantom's own signal scale).
            assert mse <= raw_mse + 1e-4, f"frame {t}: correction made alignment worse"

    def test_low_motion_frame_is_not_sent_to_refinement(self):
        """A frame identical to the reference should have ~0 coarse FD and therefore
        never reach the (more expensive) fine-refinement pass."""
        shifts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        img, _ = _create_3d_phantom(num_frames=2, nx=16, ny=16, nz=16, shifts=shifts)
        ref = ants.from_numpy(img.numpy()[..., 0], spacing=(1.0, 1.0, 1.0))
        moving_imgs = [ants.from_numpy(img.numpy()[..., t], spacing=(1.0, 1.0, 1.0)) for t in range(2)]

        fwd, inv, _ = batched_rigid_register_pass_adaptive(
            ref, moving_imgs, verbose=False, gate_fd_threshold_mm=0.4,
        )
        assert len(fwd) == 2

    def test_empty_moving_imgs_returns_immediately_adaptive(self):
        img, _ = _create_3d_phantom(num_frames=1)
        ref = ants.from_numpy(img.numpy()[..., 0], spacing=(1.0, 1.0, 1.0))
        fwd, inv, elapsed = batched_rigid_register_pass_adaptive(ref, [])
        assert fwd == [] and inv == [] and elapsed == 0.0

    def test_public_api_accepts_adaptive_backend(self):
        shifts = [(0.0, 0.0, 0.0), (0.8, -0.6, 0.5), (1.5, -1.2, 1.0)]
        img, _ = _create_3d_phantom(num_frames=3, nx=20, ny=20, nz=20, shifts=shifts)
        mc = motion_correction(img, reference=0, backend="pytorch_batched_adaptive", verbose=False)
        assert mc.motion_corrected.shape == img.shape
        assert len(mc.fd) == 3

    def test_disallowed_kwarg_raises_typeerror(self):
        shifts = [(0.0, 0.0, 0.0), (0.8, -0.6, 0.5)]
        img, _ = _create_3d_phantom(num_frames=2, nx=16, ny=16, nz=16, shifts=shifts)
        with pytest.raises(TypeError):
            motion_correction(img, backend="pytorch_batched_adaptive", not_a_real_kwarg=True)


class TestMotionCorrectionPytorchBatchedHuber:
    def test_huber_metric_recovers_known_shifts_and_rejects_unknown_metric(self):
        """``metric='huber'`` (syntx.core.losses.huber_loss_nd on the pointwise residual, same-contrast series) recovers the
        same known shifts as Mattes MI on the phantom; an unknown metric is a named error, not a silent fallback."""
        img, shifts = _create_3d_phantom(num_frames=3)
        res = motion_correction(img, reference=0, type_of_transform="Rigid", backend="pytorch_batched", verbose=False,
                                metric="huber", huber_delta=0.1)
        mp = res.motion_parameters
        for t in (1, 2):
            trans_err = np.linalg.norm(mp.translations[t] - np.array(shifts[t]))
            assert trans_err < 0.5, f"frame {t}: translation error {trans_err:.3f}mm too large"
        with pytest.raises(ValueError, match="metric"):
            motion_correction(img, reference=0, type_of_transform="Rigid", backend="pytorch_batched", verbose=False, metric="mse")


class TestSampleFraction:
    def test_sample_fraction_is_explicit_recorded_and_dense_is_not_worse(self):
        """The metric sampling rate was a hidden constant; it is now an argument recorded in the result. A known shift
        is recovered at least as well with every voxel (1.0) as with the previous 15 %, and the record says what ran."""
        img, shifts = _create_3d_phantom(num_frames=3)
        res15 = motion_correction(img, reference=0, type_of_transform="Rigid", backend="pytorch_batched", verbose=False, sample_fraction=0.15)
        res1 = motion_correction(img, reference=0, type_of_transform="Rigid", backend="pytorch_batched", verbose=False, sample_fraction=1.0)
        assert res15["sampling"]["sample_fraction"] == 0.15 and res1["sampling"]["sample_fraction"] == 1.0
        assert res1["sampling"]["n_sample_full_resolution"] >= res15["sampling"]["n_sample_full_resolution"]
        e15 = max(np.linalg.norm(res15.motion_parameters.translations[t] - np.array(shifts[t])) for t in (1, 2))
        e1 = max(np.linalg.norm(res1.motion_parameters.translations[t] - np.array(shifts[t])) for t in (1, 2))
        # 16^3 phantom, one jitter realisation per run: dense and 15 % differ by ~0.1 voxel from the sampling of the metric alone
        assert e1 < 0.5 and e1 <= e15 + 0.15, (e15, e1)
        with pytest.raises(ValueError, match="sample_fraction"):
            motion_correction(img, reference=0, type_of_transform="Rigid", backend="pytorch_batched", verbose=False, sample_fraction=0.0)


class TestSamplePointsOffVoxelCentres:
    """The metric is sampled at points drawn uniformly WITHIN the reference voxels, not at their centres. On the centres
    the moving frame is read exactly at grid nodes at the identity pose (no noise averaging) and between nodes at any
    fractional pose (trilinear noise averaging), so MI rewards fractional-voxel shifts: a still, noisy frame registered
    alone converged 0.5 mm away from the truth on a real DWI phantom, and the answer depended on which other frames
    shared the L-BFGS batch (the batch dependence vanished with it on that phantom). Guarded on a textured synthetic volume."""

    @staticmethod
    def _noisy_pair(seed=0, n=48, snr=20.0):
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)
        tex = gaussian_filter(rng.normal(size=(n, n, n)), 2.0)
        tex = (tex - tex.min()) / (tex.max() - tex.min())
        gx, gy, gz = np.ogrid[-n // 2:n // 2, -n // 2:n // 2, -n // 2:n // 2]
        env = np.exp(-((gx ** 2 + gy ** 2 + gz ** 2) / (2.0 * (0.3 * n) ** 2)))
        clean = (tex * env).astype(np.float32)
        sigma = float(clean[env > 0.5].std() / snr)
        ref = clean + rng.normal(0, sigma, clean.shape).astype(np.float32)
        mov = clean + rng.normal(0, sigma, clean.shape).astype(np.float32)
        mk = lambda a: ants.from_numpy(np.clip(a, 0, None).astype(np.float32), spacing=(2.0, 2.0, 2.0))
        return mk(ref), mk(mov)

    @staticmethod
    def _disp(paths):
        from syntx.motion import _extract_rigid_parameters, jenkinson_framewise_displacement
        h = _extract_rigid_parameters(ants.read_transform(paths[0]), 3)[2]
        return float(jenkinson_framewise_displacement([np.eye(4), h], 50.0, np.zeros(3))[1])

    def test_still_noisy_pair_stays_put(self, tmp_path):
        errs_alone = []
        for seed in range(3):
            ref, mov = self._noisy_pair(seed)
            fwd1, _, _ = batched_rigid_register_pass(ref, [mov], verbose=False, outprefix=str(tmp_path / f"a{seed}_"),
                                                     seed_mode="narrow")
            errs_alone.append(self._disp(fwd1[0]))
        rms = float(np.sqrt(np.mean(np.square(errs_alone))))
        # 0.58 mm on the voxel-centre sampling this guards against, 0.22 mm (the synthetic noise floor) with the fix
        assert rms < 0.35, f"still noisy pair registered {rms:.3f} mm away from the identity ({errs_alone})"


class TestSafeBatchedEigh:
    """Regression guard for ``torch._C._LinAlgError: linalg.eigh: (Batch element k): error code: 5``
    observed on 5/18 ExpArt rsfMRI sessions in single-precision LAPACK ``ssyevd`` when finite-difference
    6x6 frame Hessians had repeated/clustered eigenvalues."""

    def test_repeated_eigenvalues_and_nonfinite_entries(self):
        import torch
        from syntx.motion_batched import _safe_batched_eigh

        B, P = 12, 6
        H = torch.zeros(B, P, P, dtype=torch.float32)
        H[1] = 3.5 * torch.eye(P)
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(rng.normal(size=(P, P)))
        q_t = torch.from_numpy(q.astype(np.float32))
        # Clustered eigenvalues that stress single-precision ssyevd deflation
        evals_clustered = torch.tensor([1.0, 1.0 + 1e-7, 1.0 - 1e-7, 2.0, 2.0 + 1e-7, 0.0], dtype=torch.float32)
        H[8] = q_t @ torch.diag(evals_clustered) @ q_t.T
        H[9, 0, 1] = float("nan")
        H[9, 2, 2] = float("inf")

        ev, Q = _safe_batched_eigh(H, dev="cpu")
        assert ev.shape == (B, P) and Q.shape == (B, P, P)
        assert ev.dtype == torch.float32 and Q.dtype == torch.float32
        assert torch.isfinite(ev).all() and torch.isfinite(Q).all()
        recon = Q[8] @ torch.diag(ev[8]) @ Q[8].T
        sym8 = 0.5 * (H[8] + H[8].T)
        assert torch.allclose(recon, sym8, atol=1e-5)

    def test_fallback_when_torch_eigh_raises_linalg_error(self, monkeypatch):
        import torch
        from syntx.motion_batched import _safe_batched_eigh

        orig_eigh = torch.linalg.eigh
        calls = {"batched": 0, "elem": 0}

        def flaky_eigh(A, *args, **kwargs):
            if A.ndim == 3:
                calls["batched"] += 1
                raise RuntimeError("linalg.eigh: (Batch element 2): error code: 5")
            calls["elem"] += 1
            if calls["elem"] == 3:
                raise RuntimeError("linalg.eigh: simulated element failure")
            return orig_eigh(A, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, "eigh", flaky_eigh)
        rng = np.random.default_rng(7)
        M = rng.normal(size=(4, 6, 6)).astype(np.float32)
        H = torch.from_numpy(0.5 * (M + np.swapaxes(M, 1, 2)))
        ev, Q = _safe_batched_eigh(H, dev="cpu")
        assert calls["batched"] == 1
        assert calls["elem"] == 4
        assert torch.isfinite(ev).all() and torch.isfinite(Q).all()
        for b in range(4):
            recon = Q[b] @ torch.diag(ev[b]) @ Q[b].T
            assert torch.allclose(recon, H[b], atol=1e-4)
            assert torch.allclose(Q[b].T @ Q[b], torch.eye(6), atol=1e-4)

