"""Tests for the generalized overlapping-window joint encoding-bias and head-motion solver across DWI, ASL, and SlowFlow."""

from __future__ import annotations

import ants
import numpy as np
import pytest

from syntx import (
    apply_transforms,
    build_bilateral_window_references,
    build_encoding_covariates,
    motion_correct_grouped,
    motion_correction,
    shell_groups,
)
from syntx.motion_presets import MOTION_PRESETS


def _make_brain_phantom_3d(shape: tuple[int, int, int] = (32, 32, 32), spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)):
    """Create a 3D brain-like phantom with triaxial asymmetry and non-collinear structures so all 6 rigid DOFs are well-constrained."""
    N0, N1, N2 = shape
    gx, gy, gz = np.ogrid[-N0 // 2:N0 // 2, -N1 // 2:N1 // 2, -N2 // 2:N2 // 2]
    brain = np.exp(-((gx / 7.5) ** 2 + (gy / 5.5) ** 2 + (gz / 4.0) ** 2) / 2.0).astype(np.float32)
    wm = np.exp(-(((gx - 5) / 3.5) ** 2 + ((gy + 2) / 3.0) ** 2 + ((gz - 1) / 2.5) ** 2) / 2.0).astype(np.float32)
    csf = np.exp(-(((gx + 1) / 3.0) ** 2 + ((gy - 5) / 3.5) ** 2 + ((gz + 3) / 2.5) ** 2) / 2.0).astype(np.float32)
    base = 100.0 * brain + 80.0 * wm + 70.0 * csf
    img = ants.from_numpy(base.astype(np.float32), origin=(0.0, 0.0, 0.0), spacing=spacing)
    return img, brain, wm, csf


def _apply_rigid_shift_3d(img3d: ants.ANTsImage, shift_mm: np.ndarray, rot_rad: np.ndarray | None = None) -> ants.ANTsImage:
    """Apply a ground-truth rigid transform (moving -> fixed pull) to create a synthetic frame."""
    com = np.asarray(img3d.get_center_of_mass(), dtype=np.float64)
    tx = ants.create_ants_transform(transform_type="Euler3DTransform", dimension=3)
    tx.set_fixed_parameters(com.tolist())
    r = np.zeros(3, dtype=np.float64) if rot_rad is None else np.asarray(rot_rad, dtype=np.float64)
    params = [float(r[0]), float(r[1]), float(r[2]), float(-shift_mm[0]), float(-shift_mm[1]), float(-shift_mm[2])]
    tx.set_parameters(params)
    return tx.apply_to_image(img3d, reference=img3d, interpolation="linear")


class TestBuildEncodingCovariates:
    def test_partition_of_unity_and_orthonormal_basis(self):
        # 1) ASL alternating control/label
        s_asl = np.array([(-1.0) ** t for t in range(24)], dtype=np.float64)
        cov_asl = build_encoding_covariates(state_labels=s_asl, window_size=8, window_stride=4)
        assert cov_asl["E"].shape == (24, 1)
        assert cov_asl["E_surround"].shape == (22, 1)
        # Partition of unity across overlapping windows
        psi_sum = cov_asl["psi"].sum(axis=1)
        np.testing.assert_allclose(psi_sum, np.ones(24), atol=1e-6)
        # Orthonormal Q columns
        np.testing.assert_allclose(cov_asl["Q"].T @ cov_asl["Q"], np.eye(1), atol=1e-6)
        np.testing.assert_allclose(cov_asl["E"].mean(axis=0), 0.0, atol=1e-6)

        # 2) DWI bvals + bvecs
        rng = np.random.default_rng(42)
        bvals = np.array([0, 1000, 1000, 1000, 2000, 2000, 2000, 0] * 3, dtype=np.float64)
        bvecs = rng.normal(size=(24, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        bvecs[bvals == 0] = 0.0
        cov_dwi = build_encoding_covariates(bvals=bvals, bvecs=bvecs, window_size=12)
        assert cov_dwi["E"].shape[0] == 24
        assert cov_dwi["K_eff"] >= 3
        np.testing.assert_allclose(cov_dwi["psi"].sum(axis=1), np.ones(24), atol=1e-6)

        # 3) SlowFlow multi-state cycling [0, 1, 2, 3]
        states_sf = np.tile([0, 1, 2, 3], 6)
        cov_sf = build_encoding_covariates(state_labels=states_sf, window_size=8)
        assert cov_sf["E"].shape == (24, 3)
        np.testing.assert_allclose(cov_sf["psi"].sum(axis=1), np.ones(24), atol=1e-6)


class TestOverlappingWindowASL:
    def test_asl_alternating_control_label_debiasing(self):
        """ASL series with alternating control/label contrast shift + smooth drift + single spike."""
        assert "slowflow" in MOTION_PRESETS
        base_img, brain, wm, _ = _make_brain_phantom_3d()
        n_frames = 16
        s_t = np.array([(-1.0) ** t for t in range(n_frames)], dtype=np.float64)

        # True head motion: smooth drift + 1.2 mm jump at frame 8
        t_grid = np.linspace(0.0, 1.0, n_frames)
        true_tx = 0.4 * t_grid + 1.2 * (np.arange(n_frames) >= 8).astype(np.float64)
        # Spurious control-label apparent shift (+0.45 mm alternating along y) + intensity modulation
        bias_ty = 0.45 * s_t

        frames = []
        for t in range(n_frames):
            # Modulate cortical intensity slightly with control/label state
            mod_arr = base_img.numpy() + float(4.0 * s_t[t]) * brain
            mod_img = ants.from_numpy(mod_arr.astype(np.float32), origin=base_img.origin, spacing=base_img.spacing)
            shift = np.array([true_tx[t], bias_ty[t], 0.0], dtype=np.float64)
            frames.append(_apply_rigid_shift_3d(mod_img, shift))

        template4d = ants.from_numpy(
            np.stack([f.numpy() for f in frames], axis=-1),
            origin=(*base_img.origin, 0.0),
            spacing=(*base_img.spacing, 1.0),
        )

        # Without encoding_covariates, alternating 0.45 mm shift creates a 0.90 mm FD sawtooth on every step
        res_uncorrected = motion_correction(
            template4d,
            preset="bold",
            encoding_covariates=None,
            verbose=False,
        )
        # With overlapping-window encoding_covariates=s_t, the alternating sawtooth is removed from head motion
        res_debiased = motion_correction(
            template4d,
            preset="bold",
            encoding_covariates=s_t,
            window_size=8,
            verbose=False,
        )

        fd_unc = np.asarray(res_uncorrected["fd"], dtype=np.float64)
        fd_deb = np.asarray(res_debiased["fd"], dtype=np.float64)
        # Exclude frame 0 and the true jump frame 8 to measure steady-state control/label sawtooth
        non_spike = [t for t in range(1, n_frames) if t != 8]
        assert np.median(fd_deb[non_spike]) < 0.55 * np.median(fd_unc[non_spike])
        # True spike at frame 8 is preserved
        assert fd_deb[8] > 0.6
        assert res_debiased["encoding_pose_bias_mm_rad"] is not None
        assert res_debiased["encoding_pose_bias_mm_rad"]["delta_Y_mm_rad"].shape == (n_frames, 6)


class TestOverlappingWindowSlowFlow:
    def test_slowflow_cycling_states_and_shared_center(self, tmp_path):
        """SlowFlow cycling [b0, +X, +Y, +Z] with CSF contrast modulation + encoding-locked shift + shared c_0."""
        base_img, brain, wm, csf = _make_brain_phantom_3d()
        n_frames = 16
        states = np.tile([0, 1, 2, 3], n_frames // 4)
        cov_info = build_encoding_covariates(state_labels=states, window_size=8)
        E_sf = cov_info["E"]

        # True head motion: smooth drift + spike at frame 8
        t_grid = np.linspace(0.0, 1.0, n_frames)
        true_tz = 0.35 * t_grid + 1.1 * (np.arange(n_frames) >= 8).astype(np.float64)
        # State-locked apparent shift along x/y for states 1, 2, 3
        state_shift_x = np.array([0.0, 0.50, -0.35, 0.15], dtype=np.float64)[states]

        frames = []
        for t in range(n_frames):
            st = int(states[t])
            # Attenuate CSF on non-b0 states and add directional shading
            att = 1.0 if st == 0 else 0.45
            arr = 100.0 * brain + 60.0 * wm + (80.0 * att) * csf
            img_t = ants.from_numpy(arr.astype(np.float32), origin=base_img.origin, spacing=base_img.spacing)
            frames.append(_apply_rigid_shift_3d(img_t, np.array([state_shift_x[t], 0.0, true_tz[t]])))

        # Verify bilateral window references preserve state contrast
        bilateral_refs = build_bilateral_window_references(frames, encoding_covariates=E_sf, window_size=8)
        assert len(bilateral_refs) == n_frames

        template4d = ants.from_numpy(
            np.stack([f.numpy() for f in frames], axis=-1),
            origin=(*base_img.origin, 0.0),
            spacing=(*base_img.spacing, 1.0),
        )

        outprefix = str(tmp_path / "sf_")
        res_sf = motion_correction(
            template4d,
            preset="slowflow",
            encoding_covariates=E_sf,
            window_size=8,
            outprefix=outprefix,
            verbose=False,
        )

        fd_sf = np.asarray(res_sf["fd"], dtype=np.float64)
        non_spike = [t for t in range(1, n_frames) if t != 8]
        # Steady-state FD across cycling states should be low while spike at frame 8 is preserved
        assert float(np.median(fd_sf[non_spike])) < 0.35
        assert float(fd_sf[8]) > 0.55
        # The poses are in ONE common space (references averaged over the whole series, not a sliding window): the
        # slow drift and the jump are recovered, and the state-locked x shift is not reported as head motion.
        mp = np.asarray(res_sf["motion_parameters"], dtype=np.float64)
        tz_err = (mp[:, 2] - mp[:, 2].mean()) - (true_tz - true_tz.mean())
        assert float(np.sqrt(np.mean(tz_err ** 2))) < 0.05
        assert abs(float(mp[8, 2] - mp[7, 2]) - 1.1) < 0.15
        assert float(np.ptp(mp[:, 0])) < 0.15

        # Verify all written .mat transforms share the exact same FixedParameters center c_0
        fixed_centers = []
        for chain in res_sf["fwdtransforms"]:
            tx = ants.read_transform(chain[-1])
            fixed_centers.append(np.asarray(tx.fixed_parameters, dtype=np.float64))
        for c in fixed_centers[1:]:
            np.testing.assert_allclose(c, fixed_centers[0], atol=1e-5)


class TestOverlappingWindowDWI:
    def test_dwi_multishell_grouped_overlapping_window(self):
        """Multi-shell DWI (b=0, 1000) with gradient-correlated PE shift + smooth head drift."""
        base_img, brain, wm, csf = _make_brain_phantom_3d()
        bvals = np.array([0, 1000, 1000, 1000, 0, 1000, 1000, 1000, 0, 1000, 1000, 1000], dtype=np.float64)
        n_frames = len(bvals)
        rng = np.random.default_rng(7)
        bvecs = rng.normal(size=(n_frames, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        bvecs[bvals == 0] = 0.0

        t_grid = np.linspace(0.0, 1.0, n_frames)
        true_tx = 0.3 * t_grid
        # Gradient-correlated eddy translation along Y (PE axis = [0, 1, 0])
        eddy_ty = 0.55 * bvecs[:, 1] * (bvals / 1000.0)

        frames = []
        for t in range(n_frames):
            is_b0 = bvals[t] < 50
            arr = (100.0 * brain + 60.0 * wm + 80.0 * csf) if is_b0 else (55.0 * brain + 45.0 * wm + 15.0 * csf)
            img_t = ants.from_numpy(arr.astype(np.float32), origin=base_img.origin, spacing=base_img.spacing)
            frames.append(_apply_rigid_shift_3d(img_t, np.array([true_tx[t], eddy_ty[t], 0.0])))

        template4d = ants.from_numpy(
            np.stack([f.numpy() for f in frames], axis=-1),
            origin=(*base_img.origin, 0.0),
            spacing=(*base_img.spacing, 1.0),
        )
        groups = shell_groups(bvals)
        pe_cov = bvecs * (bvals[:, None] / 1000.0)

        res_dwi = motion_correct_grouped(
            template4d,
            groups,
            preset="dwi",
            pe_affine_covariates=pe_cov,
            pe_axis=(0.0, 1.0, 0.0),
            window_size=8,
            verbose=False,
        )
        fd_clean = np.asarray(res_dwi["fd"], dtype=np.float64)
        fd_raw = np.asarray(res_dwi["fd_raw"], dtype=np.float64)
        assert float(np.mean(fd_clean[1:])) < float(np.mean(fd_raw[1:]))
        assert res_dwi["encoding_pose_bias_mm_rad"] is not None


class TestOverlappingWindowBOLD:
    def test_bold_pe_respiratory_pseudomotion_separation(self):
        """BOLD EPI series with respiratory B0 oscillation along PE axis + smooth head drift + 3D head jerk."""
        base_img, _, _, _ = _make_brain_phantom_3d()
        n_frames = 20
        t_idx = np.arange(n_frames, dtype=np.float64)
        # Smooth postural drift along x and y + true 3D head jerk at frame 10
        true_tx = 0.25 * (t_idx / n_frames) + 0.90 * (t_idx >= 10).astype(np.float64)
        true_ty = 0.20 * (t_idx / n_frames) + 0.80 * (t_idx >= 10).astype(np.float64)
        # Continuous respiratory Delta B0 pseudomotion along PE axis y (0.45 mm oscillation)
        resp_b0_ty = 0.45 * np.sin(2.0 * np.pi * t_idx / 4.0)

        frames = []
        for t in range(n_frames):
            shift = np.array([true_tx[t], true_ty[t] + resp_b0_ty[t], 0.0], dtype=np.float64)
            frames.append(_apply_rigid_shift_3d(base_img, shift))

        template4d = ants.from_numpy(
            np.stack([f.numpy() for f in frames], axis=-1),
            origin=(*base_img.origin, 0.0),
            spacing=(*base_img.spacing, 1.0),
        )

        res_bold = motion_correction(
            template4d,
            preset="bold",
            pe_axis=(0.0, 1.0, 0.0),
            window_size=6,
            debias_pe_pseudomotion=True,
            verbose=False,
        )
        fd_clean = np.asarray(res_bold["fd"], dtype=np.float64)
        fd_raw = np.asarray(res_bold["fd_raw"], dtype=np.float64)
        non_spike = [t for t in range(1, n_frames) if t not in (9, 10, 11)]
        # Respiratory pseudomotion on quiet frames is strongly suppressed while true 3D spike at frame 10 is preserved
        assert float(np.median(fd_clean[non_spike])) < 0.55 * float(np.median(fd_raw[non_spike]))
        assert float(fd_clean[10]) > 0.70
        assert res_bold.get("pe_pseudomotion_debias") is not None
        assert res_bold.get("pe_b0_shift_mm") is not None


    def test_bold_default_keeps_sagittal_nodding(self):
        """Without opt-in nothing is re-attributed; with opt-in only the PE translation changes (slice and pitch never)."""
        base_img, _, _, _ = _make_brain_phantom_3d()
        n_frames = 20
        t_idx = np.arange(n_frames, dtype=np.float64)
        # sagittal nodding: slice translation (z) + pitch about the readout axis (x), oscillating like respiration
        nod_tz = 0.5 * np.sin(2.0 * np.pi * t_idx / 5.0)
        nod_rx = np.radians(0.6) * np.sin(2.0 * np.pi * t_idx / 5.0)
        frames = [_apply_rigid_shift_3d(base_img, np.array([0.0, 0.0, nod_tz[t]]), np.array([nod_rx[t], 0.0, 0.0]))
                  for t in range(n_frames)]
        template4d = ants.from_numpy(np.stack([f.numpy() for f in frames], axis=-1),
                                     origin=(*base_img.origin, 0.0), spacing=(*base_img.spacing, 1.0))
        res = motion_correction(template4d, preset="bold", pe_axis=(0.0, 1.0, 0.0), window_size=6, verbose=False)
        assert res.get("pe_pseudomotion_debias") is None and res.get("fd_raw") is None
        res_opt = motion_correction(template4d, preset="bold", pe_axis=(0.0, 1.0, 0.0), window_size=6,
                                    debias_pe_pseudomotion=True, verbose=False)
        mp = np.asarray(res["motion_parameters"], dtype=np.float64)
        mp_opt = np.asarray(res_opt["motion_parameters"], dtype=np.float64)
        # tz (index 2) and the three rotations are identical; only ty (PE) may differ
        np.testing.assert_allclose(mp_opt[:, [0, 2, 3, 4, 5]], mp[:, [0, 2, 3, 4, 5]], atol=1e-6)
        # the nodding amplitude survives in the reported trajectory
        assert float(np.ptp(mp_opt[:, 2])) > 0.6

    def test_bold_nyquist_sagittal_ripple_debiased_while_nodding_untouched(self):
        """Thick-slice Nyquist TR-to-TR sagittal ripple (lag-1 step autocorr < -0.40) is shrunk by separate_bold_pe_pseudomotion
        when asked for (debias_sagittal_nyquist=True); by default slice translation and pitch are untouched."""
        base_img, _, _, _ = _make_brain_phantom_3d()
        n_frames = 20
        alt = np.array([(-1.0) ** t for t in range(n_frames)], dtype=np.float64)
        pe_shift = 0.25 * alt
        sl_ripple = 0.20 * alt
        pitch_ripple = np.radians(0.22) * alt
        frames = [
            _apply_rigid_shift_3d(
                base_img,
                np.array([0.0, pe_shift[t], sl_ripple[t]]),
                np.array([pitch_ripple[t], 0.0, 0.0]),
            )
            for t in range(n_frames)
        ]
        template4d = ants.from_numpy(
            np.stack([f.numpy() for f in frames], axis=-1),
            origin=(*base_img.origin, 0.0),
            spacing=(*base_img.spacing, 1.0),
        )
        # default: the PE separation only; slice translation and pitch are never shrunk (real jitter alternates too)
        res_def = motion_correction(template4d, preset="bold", pe_axis=(0.0, 1.0, 0.0), window_size=8,
                                    debias_pe_pseudomotion=True, verbose=False)
        meta_def = res_def["pe_pseudomotion_debias"]
        assert meta_def is None or meta_def["wiener_shrinkage_eta"][1:] == [0.0, 0.0]
        res_opt = motion_correction(
            template4d,
            preset="bold",
            pe_axis=(0.0, 1.0, 0.0),
            window_size=8,
            debias_pe_pseudomotion=True,
            debias_sagittal_nyquist=True,
            verbose=False,
        )
        meta = res_opt["pe_pseudomotion_debias"]
        assert meta is not None
        assert meta["lag1_step_autocorr"][1] < -0.40
        assert meta["lag1_step_autocorr"][2] < -0.40
        assert meta["wiener_shrinkage_eta"][1] > 0.40
        assert meta["wiener_shrinkage_eta"][2] > 0.40
        assert float(np.mean(res_opt["fd"][1:])) < 0.40 * float(np.mean(res_opt["fd_raw"][1:]))


class TestEncodingBiasInOptimizer:
    def test_written_pose_plus_offset_is_the_optimised_total(self, tmp_path):
        """B_win is a live optimisation variable: head (written) + E_psi . B_win reproduces the encoding-locked shift."""
        base_img, brain, _, _ = _make_brain_phantom_3d()
        n_frames = 16
        s_t = np.array([(-1.0) ** t for t in range(n_frames)], dtype=np.float64)
        bias_ty = 0.45 * s_t
        frames = []
        for t in range(n_frames):
            mod = ants.from_numpy((base_img.numpy() + float(4.0 * s_t[t]) * brain).astype(np.float32),
                                  origin=base_img.origin, spacing=base_img.spacing)
            frames.append(_apply_rigid_shift_3d(mod, np.array([0.0, bias_ty[t], 0.0])))
        template4d = ants.from_numpy(np.stack([f.numpy() for f in frames], axis=-1),
                                     origin=(*base_img.origin, 0.0), spacing=(*base_img.spacing, 1.0))
        res = motion_correction(template4d, preset="asl", encoding_covariates=s_t, window_size=8,
                                outprefix=str(tmp_path / "asl_"), verbose=False)
        enc = res["encoding_pose_bias_mm_rad"]
        dY = np.asarray(enc["delta_Y_mm_rad"])
        assert dY.shape == (n_frames, 6)
        # the offset carries the alternating shift: its ty correlates with s_t with the right amplitude
        assert np.corrcoef(dY[:, 1], s_t)[0, 1] > 0.9
        amp = float(np.mean(dY[:, 1] * s_t))
        assert 0.25 < abs(amp) < 0.65
        # the written head trajectory carries (almost) none of it
        mp = np.asarray(res["motion_parameters"], dtype=np.float64)
        head_ty = mp[:, 1] - mp[:, 1].mean()
        assert abs(float(np.mean(head_ty * s_t))) < 0.1
