"""Encoding-locked systematic noise vs head motion on ground-truth simulations (syntx.motion_simulation).

Each test states the strategy that removes one kind of systematic noise and checks it against the TRUE head trajectory
(not against a low FD, which a method that flattens everything also achieves). See
``benchmarks/motion_encoding_bias_simulation.py`` for the full strategy x modality x seed table.
"""

from __future__ import annotations

import ants
import numpy as np
import pytest

from syntx import build_encoding_covariates, motion_correct_grouped, motion_correction, shell_groups
from syntx.motion_simulation import (
    _tx_from_matrix,
    evaluate_motion_estimate,
    pose_to_matrix,
    simulate_asl_series,
    simulate_bold_series,
    simulate_dwi_series,
    simulate_slowflow_series,
)


def test_simulation_conventions_recover_pure_head_motion():
    """No systematic noise: the plain single-reference estimate matches the true trajectory (sign, centre, gauge)."""
    sim = simulate_asl_series(16, seed=1, perfusion=0.0)
    ev = evaluate_motion_estimate(motion_correction(sim.image, preset="bold")["fwdtransforms"], sim)
    assert ev["translation_rms_mm"] < 0.05
    assert ev["pose_error_mm"] < 0.25
    assert all(0.8 < r < 1.2 for r in ev["jump_recovery"])


def test_chain_order_matches_ants_resampling(tmp_path):
    """``[A, B]`` in a transform list acts A first (``H = H_B @ H_A``) -- the order the grouped composition and the
    cross-group corrections (prepended to chains) rely on."""
    p0 = np.array([40.0, 50.0, 30.0])
    A = pose_to_matrix(np.r_[3, 0, 0, 0, 0, 0.5], p0)
    B = pose_to_matrix(np.r_[0, 4, 0, 0.5, 0, 0], p0)
    fa, fb = str(tmp_path / "a.mat"), str(tmp_path / "b.mat")
    ants.write_transform(_tx_from_matrix(A, p0), fa)
    ants.write_transform(_tx_from_matrix(B, p0), fb)
    x0 = np.array([30.0, 40.0, 25.0])
    y = (B @ A @ np.r_[x0, 1.0])[:3]
    arr = np.zeros((80, 100, 60), dtype=np.float32)
    i = np.round(y).astype(int)
    arr[i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2, i[2] - 1:i[2] + 2] = 1.0
    out = ants.apply_transforms(fixed=ants.from_numpy(np.zeros_like(arr)), moving=ants.from_numpy(arr), transformlist=[fa, fb]).numpy()
    g = np.indices(out.shape)
    c = np.array([(g[k] * out).sum() / out.sum() for k in range(3)])
    assert np.linalg.norm(c - x0) < 1.0


def test_integer_state_labels_are_categorical():
    st = np.tile([0, 1, 2, 3], 6)
    assert build_encoding_covariates(encoding_covariates=st)["K_eff"] == 3
    assert build_encoding_covariates(encoding_covariates=st.astype(float)[:, None])["K_eff"] == 3
    assert build_encoding_covariates(encoding_covariates=np.array([1.0, -1.0] * 8))["K_eff"] == 1


def test_asl_tag_control_bias_removed_by_encoding_projection():
    """Contrast-driven tag/control bias: the projection on s_t brings the head error to the no-perfusion floor."""
    sim = simulate_asl_series(24, seed=0, perfusion=0.10)
    base = evaluate_motion_estimate(motion_correction(sim.image, preset="bold")["fwdtransforms"], sim)
    res = motion_correction(sim.image, preset="asl", encoding_covariates=sim.encoding_covariates)
    ev = evaluate_motion_estimate(res["fwdtransforms"], sim)
    assert base["translation_rms_mm"] > 0.08            # the bias is there without the projection
    assert ev["translation_rms_mm"] < 0.4 * base["translation_rms_mm"]
    assert ev["translation_rms_mm"] < 0.05
    assert ev["fd_error_mm"] < 0.5 * base["fd_error_mm"]
    assert all(0.75 < r < 1.3 for r in ev["jump_recovery"])
    assert res["encoding_pose_bias_mm_rad"]["delta_Y_mm_rad"].shape == (24, 6)


def test_slowflow_state_locked_offsets_removed_by_preset():
    """[b0, +X, +Y, +Z] with flow-encoding contrast and a state-locked eddy shift: the slowflow preset (state-matched
    references + projection on the categorical state labels) keeps the eddy shift out of the head trajectory."""
    sim = simulate_slowflow_series(6, seed=0, eddy_mm=0.4)
    base = evaluate_motion_estimate(motion_correction(sim.image, preset="bold")["fwdtransforms"], sim)
    ev = evaluate_motion_estimate(
        motion_correction(sim.image, preset="slowflow", encoding_covariates=sim.state_labels)["fwdtransforms"], sim)
    assert base["translation_rms_mm"] > 0.15
    assert ev["translation_rms_mm"] < 0.08
    assert ev["fd_error_mm"] < 0.5 * base["fd_error_mm"]
    assert all(0.75 < r < 1.3 for r in ev["jump_recovery"])


def test_dwi_grouped_eddy_applied_and_head_motion_recovered():
    """DWI: the eddy displacement (odd in g) is modelled and applied (pe_affine), the head trajectory is recovered in
    the anchor space through the composed chain (crosses + cross-group correction + own rigid)."""
    sim = simulate_dwi_series(24, seed=0, eddy_mm_per_unit=0.6)
    bv, bvec = sim.extra["bvals"], sim.extra["bvecs"]
    res = motion_correct_grouped(sim.image, shell_groups(bv), preset="dwi", pe_affine_covariates=bvec * (bv[:, None] / 1000.0),
                                 pe_axis=sim.pe_axis, cross_group_debias=True)
    ev = evaluate_motion_estimate(res["fwdtransforms"], sim)
    single = evaluate_motion_estimate(motion_correction(sim.image, preset="bold")["fwdtransforms"], sim)
    assert ev["translation_rms_mm"] < 0.12
    assert ev["translation_rms_mm"] < 0.5 * single["translation_rms_mm"]   # the single reference absorbs the eddy shift
    eddy_err = float(np.sqrt(np.mean((np.asarray(res["pe_affine"]["a"])[:, 0] - sim.eddy[:, 0]) ** 2)))
    assert eddy_err < 0.2
    assert res["cross_group_offset"] is not None


def test_bold_respiratory_pe_shift_separated_without_touching_nodding():
    """Respiratory B0 shift along PE (real displacement): opt-in separation from the REPORTED head motion; the real
    nodding (z translation + pitch) and every rotation are left exactly as registered."""
    sim = simulate_bold_series(48, seed=0, resp_mm=0.4, nod_mm=0.5, nod_deg=0.5)
    res0 = motion_correction(sim.image, preset="bold")
    res1 = motion_correction(sim.image, preset="bold", pe_axis=sim.pe_axis, debias_pe_pseudomotion=True, slice_axis=2)
    mp0 = np.asarray(res0["motion_parameters"], dtype=float)
    mp1 = np.asarray(res1["motion_parameters"], dtype=float)
    true_pe = sim.head_pose[:, :3] @ sim.pe_axis
    err0 = np.std((mp0[:, :3] @ sim.pe_axis) - true_pe)
    err1 = np.std((mp1[:, :3] @ sim.pe_axis) - true_pe)
    assert err1 < 0.5 * err0
    # only the PE translation changed: reported rotations identical, slice translation identical
    np.testing.assert_allclose(mp1[:, 3:], mp0[:, 3:], atol=1e-6)
    np.testing.assert_allclose(mp1[:, 2], mp0[:, 2], atol=1e-6)


def test_encoding_references_inclusive_by_default_and_leave_one_out_on_request():
    """Discrete states: the inclusive state mean is shared by all frames of the state (a common space). The DWI
    predictor is leave-one-out: a frame gets no weight in its own reference."""
    from syntx import build_bilateral_window_references, encoding_reference_predictor

    rng = np.random.default_rng(0)
    frames = [ants.from_numpy(rng.normal(size=(6, 6, 6)).astype(np.float32)) for _ in range(8)]
    states = np.tile([0, 1, 2, 3], 2)
    refs = build_bilateral_window_references(frames, encoding_covariates=states)
    np.testing.assert_allclose(refs[0].numpy(), refs[4].numpy(), atol=1e-4)            # one reference per state
    np.testing.assert_allclose(refs[0].numpy(), 0.5 * (frames[0].numpy() + frames[4].numpy()), atol=0.08)  # ~1 % kernel leak
    loo = build_bilateral_window_references(frames, encoding_covariates=states, exclude_self=True)
    np.testing.assert_allclose(loo[0].numpy(), frames[4].numpy(), atol=0.12)             # the other state-0 frame (+ leak)
    pred = encoding_reference_predictor(rng.normal(size=(8, 5)))(frames, group_index=1, frame_indices=list(range(8)))
    assert len(pred) == 8 and not np.allclose(pred[0].numpy(), frames[0].numpy())
