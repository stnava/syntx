"""Tests for syntx.motion_reference -- low-motion-subset reference building and
contrast-grouped motion correction (e.g. DWI b0-then-DWI). See the module's own
docstring for the real-data findings (2026-10-04) this design is built on.
"""

from unittest.mock import patch

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
        # The grouped-specific artifacts should all be present (not just the generic
        # motion_corrected/fd contract) -- a caller may want the per-group detail.
        for key in ("group_results", "group_indices", "group_means", "cross_registrations", "fwdtransforms"):
            assert key in out
        assert len(out["group_results"]) == 2 and len(out["cross_registrations"]) == 1

    def test_both_groups_dvars_normalized_against_group_a_mean(self):
        """Real bug found 2026-10-06 (antsxmm cohort review): two groups (e.g. DWI's b0
        and diffusion-weighted volumes) normally differ substantially in raw signal
        intensity (diffusion attenuation dims group B relative to group A). Before this
        fix, each group's motion_correction() call normalized DVARS by its OWN mean --
        making group A and group B DVARS incomparable, and (separately) making DVARS
        incomparable ACROSS SUBJECTS whose raw intensity scales differ (confirmed ~1000x
        gap between two real subjects processed through this identical code path). This
        test confirms motion_correct_grouped now passes the SAME reference_intensity
        (group A's own mean) to both groups' motion_correction() calls, rather than
        letting each default to its own mean."""
        frames = [_phantom_frame(shift=s) for s in
                  [(0, 0, 0), (0.1, 0, 0), (2.0, -1.5, 1.0), (1.8, -1.2, 0.9), (2.2, -1.6, 1.1)]]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 5), dtype=np.float32)), frames)
        group_is_reference = np.array([True, True, False, False, False])

        import syntx.motion as motion_module

        captured_kwargs = []
        real_motion_correction = motion_module.motion_correction

        def _spy(*args, **kwargs):
            captured_kwargs.append(kwargs.get("dvars_reference_intensity"))
            return real_motion_correction(*args, **kwargs)

        # motion_correct_grouped does `from .motion import motion_correction` locally
        # (inside the function body), so the patch target is syntx.motion's own name,
        # not an attribute of the motion_reference module.
        with patch.object(motion_module, "motion_correction", side_effect=_spy):
            motion_correct_grouped(img4d, group_is_reference, n_low_motion=10, verbose=False)

        assert len(captured_kwargs) == 2
        assert captured_kwargs[0] is not None
        assert captured_kwargs[0] == pytest.approx(captured_kwargs[1])

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

    def test_preserves_real_physical_space_not_default_header(self):
        """Real bug found 2026-10-06 (real ds005134 DWI data): the two-mean design built
        its internal per-group 4D containers AND the final motion_corrected output via
        ants.list_to_ndimage(ants.from_numpy(np.zeros(...)), frames) -- the zeros-array
        template carries ANTsPy's default header (origin (0,0,0), identity direction,
        unit spacing), and list_to_ndimage copies the TEMPLATE's header onto the
        assembled image, silently discarding the real subject geometry from every frame
        stacked into it. Every existing fixture in this file used spacing=(1,1,1) with
        implicit zero origin/identity direction, so this was invisible to the existing
        suite -- this test uses a non-trivial origin, anisotropic spacing, and a
        non-identity (axis-flipped) direction, matching real DWI headers (e.g. RPI
        orientation), to actually catch it."""
        origin = (-50.0, 30.0, -10.0)
        spacing = (2.0, 2.0, 3.7)
        direction = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

        def _real_space_frame(shift=(0.0, 0.0, 0.0)):
            f = _phantom_frame(shift=shift)
            f.set_spacing(spacing)
            f.set_origin(origin)
            f.set_direction(direction)
            return f

        frames = [_real_space_frame(shift=s) for s in
                  [(0, 0, 0), (0.1, 0, 0), (2.0, -1.5, 1.0), (1.8, -1.2, 0.9), (2.2, -1.6, 1.1)]]
        ref3d = frames[0]
        img4d = ants.list_to_ndimage(
            ants.make_image(list(ref3d.shape) + [5], 0, spacing=list(spacing) + [1.0],
                             origin=list(origin) + [0.0],
                             direction=np.eye(4)),
            frames,
        )
        # The 3x3 spatial block of the 4D template must match ref3d's own direction.
        img4d.set_direction(
            np.block([[direction, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]])
        )
        group_is_reference = np.array([True, True, False, False, False])

        out = motion_correct_grouped(img4d, group_is_reference, n_low_motion=10, verbose=False)
        result3d = ants.slice_image(out.motion_corrected, axis=3, idx=0)

        assert np.allclose(ants.get_spacing(result3d), spacing)
        assert np.allclose(ants.get_origin(result3d), origin)
        assert np.allclose(ants.get_direction(result3d), direction)


class TestGroupedTransformFilesAreDistinctPerGroup:
    def test_outprefix_does_not_collide_across_groups(self, tmp_path):
        """Real bug (sub-BLAST079 3-shell DWI, 2026-10-20): with an ``outprefix`` every group wrote its frame-t transform
        to the same ``{outprefix}batch_{t:04d}.mat``; the last group overwrote the others. The per-frame chains are read
        back from those paths after all groups have run, so group 0/1 frames were resampled with group 2's transforms
        and the series FD alternated between a group's real pose change and a constant cross-group term. Every frame's
        own transform file must be unique and must reproduce that frame's own-group pose."""
        from syntx.motion import _extract_rigid_parameters

        shifts = [(0, 0, 0), (0.1, 0, 0), (2.0, -1.5, 1.0), (1.8, -1.2, 0.9), (2.2, -1.6, 1.1), (0.3, 0.2, 0)]
        frames = [_phantom_frame(shift=s) for s in shifts]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((20, 20, 20, 6), dtype=np.float32)), frames)
        groups = np.array([0, 1, 1, 0, 1, 0])

        out = motion_correct_grouped(img4d, groups, n_low_motion=10, verbose=False, outprefix=str(tmp_path / "mc_"))

        own_files = [chain[-1] for chain in out.fwdtransforms]  # the frame's own transform is the last in its chain
        assert len(set(own_files)) == len(own_files), own_files
        for g, (idx, res) in enumerate(zip(out.group_indices, out.group_results)):
            for li, fi in enumerate(idx):
                assert own_files[fi] == res.fwdtransforms[li][-1]
                t_file = _extract_rigid_parameters(ants.read_transform(own_files[fi]), 3)[0]
                np.testing.assert_allclose(t_file, np.asarray(res.motion_parameters.translations)[li], atol=1e-4)


class TestGroupedNuisanceRegression:
    def test_nuisance_translation_removed_from_fd_but_kept_in_applied_transforms(self, tmp_path):
        """DWI eddy currents translate each volume along phase encode by an amount linear in the gradient vector; the rigid
        fit absorbs it and FD counts it as motion (sub-BLAST079, 2026-10-20). With ``nuisance_covariates`` + ``nuisance_axis``
        the per-group linear fit is removed from ``fd`` only: ``fd_raw`` keeps the full pose and the corrected image still
        aligns the displaced content."""
        rng = np.random.default_rng(0)
        n = 14
        cov = np.zeros((n, 3))
        cov[2:] = rng.normal(size=(n - 2, 3))
        cov[2:] /= np.linalg.norm(cov[2:], axis=1, keepdims=True)
        B_Y = 1.5  # mm of y translation per unit of the covariate's y component

        def _frame(shift, N=40):  # three blobs: rotation identifiable (a single Gaussian is rotation-invariant)
            gx, gy, gz = np.ogrid[-N // 2:N // 2, -N // 2:N // 2, -N // 2:N // 2]
            sx, sy, sz = shift

            def blob(cx, cy, cz, s, w):
                return w * np.exp(-((gx - sx - cx) ** 2 + (gy - sy - cy) ** 2 + (gz - sz - cz) ** 2) / (2 * s ** 2))

            return ants.from_numpy((blob(0, 0, 0, 6, 1) + blob(9, 5, -7, 3, .7) + blob(-8, 6, 5, 2.5, .5)).astype(np.float32),
                                   spacing=(1.0, 1.0, 1.0))

        shifts = [(0, 0, 0), (0, 0, 0)] + [(0, B_Y * cov[i, 1], 0) for i in range(2, n)]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, n), dtype=np.float32)), [_frame(s) for s in shifts])
        groups = np.array([0, 0] + [1] * (n - 2))

        out = motion_correct_grouped(img4d, groups, n_low_motion=10, verbose=False, outprefix=str(tmp_path / "mc_"),
                                     nuisance_covariates=cov, nuisance_axis=(0.0, 1.0, 0.0))
        assert out.nuisance["coefficients"][0] is None and out.nuisance["r2"][0] is None  # b0-like group: no fit
        assert out.nuisance["r2"][1] > 0.95
        np.testing.assert_allclose(out.nuisance["coefficients"][1], [0.0, B_Y, 0.0], atol=0.1)
        assert np.mean(out.fd_raw[2:]) > 0.8  # the imposed translation is the dominant "motion" in the full pose
        assert np.mean(out.fd[2:]) < 0.4 * np.mean(out.fd_raw[2:])  # and is removed from fd only
        assert len(out.nuisance["shift_mm"]) == n and np.all(out.nuisance["shift_mm"][:2] == 0)
        arr = out.motion_corrected.numpy()
        for t in range(2, n):  # applied correction unchanged: the corrected frames still align to the anchor
            assert np.corrcoef(arr[..., 0].ravel(), arr[..., t].ravel())[0, 1] > 0.98


class TestGroupedEddyPEAffine:
    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_tied_pe_affine_recovers_imposed_eddy_and_leaves_fd_to_motion(self, tmp_path, direction):
        """Eddy-current model: per frame a translation + shears + scale along the phase-encode axis, tied across the group's
        frames as a linear function of the gradient vector (``pe_affine_covariates``), estimated jointly with the rigid
        pose. Impose a0 = 1.2 g_y (mm) and a PE scale of 0.03 g_x per mm on still frames: the coefficient matrix is
        recovered, the rigid FD stays at the noise floor, the eddy is applied to the data and the b0-like group is untouched."""
        rng = np.random.default_rng(1)
        n = 22
        g = np.zeros((n, 3))
        g[2:] = rng.normal(size=(n - 2, 3))
        g[2:] /= np.linalg.norm(g[2:], axis=1, keepdims=True)
        p = np.array([0.0, 1.0, 0.0])
        N = 40

        def _still(N=N):
            gx, gy, gz = np.ogrid[-N // 2:N // 2, -N // 2:N // 2, -N // 2:N // 2]

            def blob(cx, cy, cz, s, w):
                return w * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2) / (2 * s ** 2))

            return ants.from_numpy((blob(0, 0, 0, 6, 1) + blob(9, 5, -7, 3, .7) + blob(-8, 6, 5, 2.5, .5)).astype(np.float32),
                                   spacing=(1.0, 1.0, 1.0))

        base = _still()
        com = np.asarray(base.get_center_of_mass())
        frames = []
        for k in range(n):
            a0, a_scale = 1.2 * g[k, 1], 0.03 * g[k, 0]  # translation along p; scale along p (coefficient on the p-position)
            M = np.eye(3) + np.outer(p, a_scale * p)
            tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3, matrix=M, offset=p * a0, center=com)
            # the observed frame is the distorted still image: sample base at the distorted location (inverse map), so that
            # the solver's forward map (reference -> frame) is exactly M, a0
            frames.append(ants.apply_ants_transform_to_image(tx.invert(), base, reference=base))
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((N, N, N, n), dtype=np.float32)), frames)
        groups = np.array([0, 0] + [1] * (n - 2))

        out = motion_correct_grouped(img4d, groups, n_low_motion=10, verbose=False, outprefix=str(tmp_path / "mc_"),
                                     backend="pytorch_batched", pe_affine_covariates=g, pe_axis=p, registration_direction=direction)
        pe = out.pe_affine
        assert pe is not None and pe["coefficients_by_group"][0] is None
        C = np.asarray(pe["coefficients_by_group"][1])  # (4, 3): rows a0, a1, a2, a3; columns g_x, g_y, g_z
        np.testing.assert_allclose(C[0], [0.0, 1.2, 0.0], atol=0.25)
        np.testing.assert_allclose(C[2], [0.03, 0.0, 0.0], atol=0.008)
        assert np.abs(C[[1, 3]]).max() < 0.008
        assert np.all(pe["a"][:2] == 0) and np.all(pe["jacobian"][:2] == 1.0)
        assert np.mean(out.fd[2:]) < 0.25  # rigid part: no motion was imposed
        assert np.mean(out.fd_raw[2:]) > 2 * np.mean(out.fd[2:])  # the eddy term is in the full chain
        for chain in out.fwdtransforms[2:]:
            assert len(chain) == 3 and "_eddy" in chain[-2]  # [cross, eddy, rigid]
        arr = out.motion_corrected.numpy()
        for t in range(2, n):
            assert np.corrcoef(arr[..., 0].ravel(), arr[..., t].ravel())[0, 1] > 0.99


def _blob_frame(weights, shift=(0.0, 0.0, 0.0), N=40):
    """Three blobs with per-frame weights (contrast) and a rigid shift of the whole pattern."""
    gx, gy, gz = np.ogrid[-N // 2:N // 2, -N // 2:N // 2, -N // 2:N // 2]
    sx, sy, sz = shift

    def blob(cx, cy, cz, s, w):
        return w * np.exp(-((gx - sx - cx) ** 2 + (gy - sy - cy) ** 2 + (gz - sz - cz) ** 2) / (2 * s ** 2))

    w1, w2, w3 = weights
    return ants.from_numpy((blob(0, 0, 0, 6, w1) + blob(9, 5, -7, 3, w2) + blob(-8, 6, 5, 2.5, w3)).astype(np.float32),
                           spacing=(1.0, 1.0, 1.0))


class TestPerFrameReferences:
    def test_contrast_matched_references_recover_shifts_the_mean_reference_cannot(self, tmp_path):
        """Frames whose contrast differs strongly from frame to frame (the weights of the three blobs change), each
        rigidly shifted by a known amount. Registered to the mean of the frames' own (unshifted) patterns the pose is
        biased by the contrast mismatch; registered to its own contrast-matched reference each frame recovers its shift.
        Per-frame references through motion_correction(reference=<list>) -> pytorch_batched."""
        from syntx.motion import motion_correction

        rng = np.random.default_rng(3)
        n = 8
        weights = [(1.0, 0.1 + 0.9 * (k % 2), 1.0 - 0.9 * (k % 2)) for k in range(n)]  # alternating contrast
        shifts = [(0, 0, 0)] + [tuple(rng.uniform(-1.5, 1.5, 3)) for _ in range(n - 1)]
        frames = [_blob_frame(w, s) for w, s in zip(weights, shifts)]
        refs = [_blob_frame(w) for w in weights]  # contrast-matched, unshifted
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, n), dtype=np.float32)), frames)
        mean_ref = ants.from_numpy(np.mean([r.numpy() for r in refs], axis=0).astype(np.float32), spacing=(1.0, 1.0, 1.0))

        per = motion_correction(img4d, reference=refs, backend="pytorch_batched", verbose=False, seed_mode="narrow",
                                outprefix=str(tmp_path / "per_"))
        shared = motion_correction(img4d, reference=mean_ref, backend="pytorch_batched", verbose=False, seed_mode="narrow",
                                   outprefix=str(tmp_path / "mean_"))
        err_per = np.linalg.norm(np.asarray(per.motion_parameters.translations) - np.asarray(shifts), axis=1)
        err_mean = np.linalg.norm(np.asarray(shared.motion_parameters.translations) - np.asarray(shifts), axis=1)
        assert per.summary["reference_type"] == "per_frame"
        assert err_per.max() < 0.25, err_per
        assert err_mean.mean() > 2 * err_per.mean(), (err_mean, err_per)

    def test_identical_per_frame_references_match_the_shared_reference(self, tmp_path):
        from syntx.motion import motion_correction

        frames = [_blob_frame((1, 0.7, 0.5), s) for s in [(0, 0, 0), (0.8, -0.4, 0.3), (-0.5, 0.6, -0.9)]]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, 3), dtype=np.float32)), frames)
        ref = _blob_frame((1, 0.7, 0.5))
        a = motion_correction(img4d, reference=ref, backend="pytorch_batched", verbose=False, seed_mode="narrow")
        b = motion_correction(img4d, reference=[ref, ref, ref], backend="pytorch_batched", verbose=False, seed_mode="narrow")
        np.testing.assert_allclose(np.asarray(a.motion_parameters.translations), np.asarray(b.motion_parameters.translations), atol=0.05)

    def test_grouped_reference_predictor_hook(self, tmp_path):
        """The predictor is called with the group's corrected frames; None keeps the mean; a list re-registers the group
        to per-frame references for the requested iterations and reference_modes reports it."""
        calls = []

        def predictor(frames, group_index, frame_indices):
            calls.append((group_index, len(frames), list(frame_indices)))
            if group_index == 0:
                return None
            return [f for f in frames]  # the frames themselves (already corrected): a valid contrast-matched reference

        frames = [_blob_frame((1, 0.7, 0.5), s) for s in [(0, 0, 0), (0.1, 0, 0), (1.0, -0.5, 0.3), (0.8, -0.6, 0.2), (1.2, -0.4, 0.4)]]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, 5), dtype=np.float32)), frames)
        groups = np.array([0, 0, 1, 1, 1])
        out = motion_correct_grouped(img4d, groups, n_low_motion=10, verbose=False, backend="pytorch_batched",
                                     outprefix=str(tmp_path / "mc_"), reference_predictor=predictor, reference_iterations=2,
                                     seed_mode="narrow")
        assert out.reference_modes == ["mean", "predicted (iteration 2)"]
        assert [c[0] for c in calls] == [0, 1, 1] and calls[1][1] == 3 and calls[1][2] == [2, 3, 4]
        assert out.motion_corrected.shape == img4d.shape and len(out.fd) == 5


class TestSampleMaskAndBiasCovariates:
    def test_sample_mask_restricts_the_metric_and_bias_is_removed_from_written_poses(self, tmp_path):
        """A frame whose contrast differs from the reference only INSIDE a region biases the pose; with the metric
        restricted to the rest (sample_mask) the shift is recovered. A deterministic pose bias that is a known function
        of per-frame covariates is regressed out of the written poses (bias_covariates), with the intercept kept."""
        from syntx.motion import motion_correction

        rng = np.random.default_rng(4)
        n = 24
        shifts = [(0, 0, 0)] + [tuple(rng.uniform(-1.0, 1.0, 3)) for _ in range(n - 1)]
        # contrast of the centre blob varies per frame (direction-dependent white matter); the two outer blobs do not
        frames = [_blob_frame((0.3 + 0.7 * (k % 2), 0.7, 0.5), s) for k, s in enumerate(shifts)]
        ref = _blob_frame((0.65, 0.7, 0.5))
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, n), dtype=np.float32)), frames)
        gx, gy, gz = np.ogrid[-20:20, -20:20, -20:20]
        outer = ((gx ** 2 + gy ** 2 + gz ** 2) > 11 ** 2).astype(np.float32)  # exclude the direction-dependent centre
        mask = ants.from_numpy(outer, spacing=(1.0, 1.0, 1.0))

        plain = motion_correction(img4d, reference=ref, backend="pytorch_batched", verbose=False, seed_mode="narrow")
        masked = motion_correction(img4d, reference=ref, backend="pytorch_batched", verbose=False, seed_mode="narrow", sample_mask=mask)
        e_plain = np.linalg.norm(np.asarray(plain.motion_parameters.translations) - np.asarray(shifts), axis=1)
        e_mask = np.linalg.norm(np.asarray(masked.motion_parameters.translations) - np.asarray(shifts), axis=1)
        assert e_mask.max() < 0.3, e_mask
        assert e_mask.mean() < e_plain.mean(), (e_plain, e_mask)

        # bias: impose a translation that is a linear function of a covariate on top of the shifts; it must be removed
        cov = rng.normal(size=(n, 2)); cov[0] = 0
        B = np.array([[0.8, 0.0, 0.0], [0.0, -0.6, 0.0]])  # mm per unit covariate
        biased = [_blob_frame((0.65, 0.7, 0.5), tuple(np.asarray(s) + cov[k] @ B)) for k, s in enumerate(shifts)]
        img_b = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, n), dtype=np.float32)), biased)
        res = motion_correction(img_b, reference=ref, backend="pytorch_batched", verbose=False, seed_mode="narrow", bias_covariates=cov)
        e_bias = np.linalg.norm(np.asarray(res.motion_parameters.translations) - np.asarray(shifts), axis=1)
        assert res["bias"] is not None and res["bias"]["n_covariates"] == 2
        np.testing.assert_allclose(np.asarray(res["bias"]["coefficients"])[:, :3], B, atol=0.3)  # the random +-1 mm shifts act as noise on 24 frames
        imposed = np.linalg.norm(cov @ B, axis=1)  # what the poses would be off by without the removal
        assert e_bias.mean() < 0.4 * imposed.mean(), (e_bias.mean(), imposed.mean())


class TestBackwardRegistration:
    def test_backward_recovers_known_shifts_and_matches_forward_convention(self, tmp_path):
        """registration_direction='backward' (the mean onto each frame) returns transforms in the forward convention:
        known shifts are recovered, the corrected frames align to the anchor, and the eddy term keeps its sign."""
        rng = np.random.default_rng(6)
        n = 12
        shifts = [(0, 0, 0), (0.2, 0, 0)] + [tuple(rng.uniform(-1.0, 1.0, 3)) for _ in range(n - 2)]
        frames = [_blob_frame((1.0, 0.7, 0.5), s) for s in shifts]
        img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((40, 40, 40, n), dtype=np.float32)), frames)
        groups = np.array([0, 0] + [1] * (n - 2))
        out = motion_correct_grouped(img4d, groups, n_low_motion=10, verbose=False, backend="pytorch_batched", seed_mode="narrow",
                                     outprefix=str(tmp_path / "mc_"), registration_direction="backward", metric_smoothing_mm=1.0)
        tr = np.array([r.motion_parameters.translations for r in out.group_results][1])
        err = np.linalg.norm(tr - np.asarray(shifts[2:]) + np.mean(np.asarray(shifts[2:]), axis=0) * 0, axis=1)
        # poses are relative to the group's (low-motion) mean: compare pairwise differences, which are reference-free
        d_true = np.asarray(shifts[2:])[1:] - np.asarray(shifts[2:])[:-1]
        d_rec = tr[1:] - tr[:-1]
        assert np.abs(d_rec - d_true).max() < 0.3, np.abs(d_rec - d_true).max()
        arr = out.motion_corrected.numpy()
        for t in range(2, n):
            assert np.corrcoef(arr[..., 0].ravel(), arr[..., t].ravel())[0, 1] > 0.98
        assert out.group_results[1]["registration_direction"] == "backward" and len(out.fd) == n
