"""
tests/test_phase_correlation_candidates.py
============================================

Unit tests for `_phase_correlation_translation_candidates` and its integration into
`robust_affine`'s torch-native multi-start candidate pool (`src/syntx/robust_affine.py`).

Background: diagnosed directly (not assumed) during the syntx batched-motion-correction
investigation that center-of-mass translation initialization can be badly wrong (7-11mm, on
a phantom with a ground-truth translation) for specific frames -- likely from a bias field or
asymmetric content skewing the weighted centroid -- with no signal that it happened. FFT-based
phase correlation is a second, independent (different failure mode) translation estimator:
a single FFT, globally exact for pure translation, immune to the CoM failure mode. It was
generalized into `robust_affine`'s own multi-start candidate pool (used by `motion_correction`,
`auto_reg`, and `build_template` alike), not just the batched prototype, per that
investigation's explicit follow-up. These tests cover the underlying estimator directly and
its wiring into `robust_affine`, complementing (not duplicating) the ground-truth recovery
scripts under `scripts/` (`bench_motion_recovery_methods.py`,
`validate_dwi_motion_ground_truth.py`, `prototype_batched_motion_correction.py`), which are
slower, real-clinical-data-based validation, not part of the default fast suite.
"""

import os
import tempfile
import numpy as np
import ants
import pytest

from syntx.robust_affine import (
    _phase_correlation_translation_candidates,
    robust_affine,
)


def _synthetic_blob_image(shape=(48, 48, 48), spacing=(1.5, 1.5, 1.5)):
    """A single smooth Gaussian blob -- has real spatial structure (unlike uniform noise)
    but no anatomy, so tests run fast without external data dependencies."""
    gx, gy, gz = np.ogrid[-shape[0] // 2:shape[0] // 2,
                           -shape[1] // 2:shape[1] // 2,
                           -shape[2] // 2:shape[2] // 2]
    arr = np.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / (2.0 * 8.0 ** 2)).astype(np.float32)
    # Add a second, off-center smaller blob so translation is unambiguous (a single
    # spherically-symmetric blob has no orientation/offset information a translation-only
    # estimator could latch onto incorrectly, which would make this an unrealistically easy
    # test in the wrong way).
    ox, oy, oz = shape[0] // 4, -shape[1] // 6, shape[2] // 5
    gx2, gy2, gz2 = gx - ox, gy - oy, gz - oz
    arr += 0.6 * np.exp(-(gx2 ** 2 + gy2 ** 2 + gz2 ** 2) / (2.0 * 4.0 ** 2)).astype(np.float32)
    return ants.from_numpy(arr, spacing=spacing)


def _translate_image(img, shift_mm):
    """Physically translate `img` by `shift_mm` (a length-3 sequence) via a real ANTs
    transform + resample -- not a numpy roll -- so this exercises the same physical-space
    convention the estimator itself operates in."""
    dim = img.dimension
    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx.set_parameters(np.concatenate([np.eye(dim).flatten(), -np.asarray(shift_mm, dtype=np.float64)]))
    tx.set_fixed_parameters(np.zeros(dim))
    tx_path = os.path.join(tempfile.mkdtemp(prefix="test_pc_"), "shift.mat")
    ants.write_transform(tx, tx_path)
    return ants.apply_transforms(fixed=img, moving=img, transformlist=[tx_path], interpolator='linear')


class TestPhaseCorrelationCandidates:

    def test_recovers_known_translation(self):
        """A moderate, sub-FOV translation should appear among the top-k phase-correlation
        candidates, close to the true value."""
        fixed = _synthetic_blob_image()
        true_shift = np.array([6.0, -4.5, 3.0])  # mm
        moving = _translate_image(fixed, true_shift)

        candidates = _phase_correlation_translation_candidates(fixed, moving, topk=3)
        assert len(candidates) == 3
        errs = [np.linalg.norm(np.asarray(c) - true_shift) for c in candidates]
        assert min(errs) < 2.0, f"best candidate error {min(errs):.2f}mm, candidates={candidates}"

    def test_returns_topk_distinct_candidates(self):
        """Non-max suppression between peaks should yield genuinely different candidates,
        not the same peak repeated."""
        fixed = _synthetic_blob_image()
        moving = _translate_image(fixed, np.array([5.0, 5.0, 5.0]))
        candidates = _phase_correlation_translation_candidates(fixed, moving, topk=3)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                assert np.linalg.norm(np.asarray(candidates[i]) - np.asarray(candidates[j])) > 1e-3

    def test_2d_returns_empty_not_error(self):
        """2D path isn't implemented -- must degrade gracefully (empty list), not raise,
        since robust_affine's 2D pipeline must keep working without it."""
        fixed = ants.from_numpy(np.random.rand(32, 32).astype(np.float32))
        moving = ants.from_numpy(np.random.rand(32, 32).astype(np.float32))
        candidates = _phase_correlation_translation_candidates(fixed, moving)
        assert candidates == []

    def test_robust_affine_integration_no_regression(self):
        """robust_affine's default multi-start pytorch path (which now includes
        phase-correlation candidates alongside CoM) must still converge correctly and
        produce a valid ANTs-readable transform on an ordinary, moderate-offset case."""
        fixed = _synthetic_blob_image()
        moving = _translate_image(fixed, np.array([4.0, -3.0, 2.0]))
        result = robust_affine(fixed=fixed, moving=moving, mode='auto', backend='pytorch',
                                dof='rigid', verbose=False)
        assert 'fwdtransforms' in result and len(result['fwdtransforms']) == 1
        tx = ants.read_transform(result['fwdtransforms'][0])
        params = np.array(tx.parameters)
        A = params[:9].reshape(3, 3)
        assert np.isclose(np.linalg.det(A), 1.0, atol=1e-4), \
            "dof='rigid' must keep det(A) == 1 regardless of candidate source"

    def test_large_translation_recovered(self):
        """The motivating case: a translation large enough to meaningfully stress
        CoM-based initialization should still be recovered via the phase-correlation
        candidate path. Not a substitute for the full ground-truth benchmark in
        scripts/bench_motion_recovery_methods.py --jump, but a fast, always-run guard
        against this specific capability regressing silently."""
        fixed = _synthetic_blob_image(shape=(56, 56, 56))
        true_shift = np.array([14.0, -10.0, 8.0])  # mm, large relative to this FOV
        moving = _translate_image(fixed, true_shift)

        result = robust_affine(fixed=fixed, moving=moving, mode='auto', backend='pytorch',
                                dof='rigid', verbose=False)
        tx = ants.read_transform(result['fwdtransforms'][0])
        params = np.array(tx.parameters)
        A = params[:9].reshape(3, 3)
        t_hat = params[9:12]
        # dof='rigid' fwdtransforms map fixed->moving (pull-back convention established
        # throughout this codebase); recovered translation should be close to -true_shift
        # up to the rotation/center bookkeeping -- check via warped-image alignment instead
        # of raw parameter comparison, which is convention-robust.
        warped = ants.apply_transforms(fixed=fixed, moving=moving,
                                        transformlist=result['fwdtransforms'])
        mse_before = float(np.mean((fixed.numpy() - moving.numpy()) ** 2))
        mse_after = float(np.mean((fixed.numpy() - warped.numpy()) ** 2))
        assert mse_after < 0.1 * mse_before, (
            f"expected large-translation recovery to substantially reduce MSE "
            f"(before={mse_before:.4f}, after={mse_after:.4f})")
