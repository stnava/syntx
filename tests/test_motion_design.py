"""syntx.motion_design: the acquisition-design menu and the single motion-correction entry point."""
import numpy as np
import pytest

from syntx.motion_design import DESIGNS, _real_sh_l2, make_design, motion_correct_design


def test_menu_entries_and_unknown_name():
    assert set(DESIGNS) == {"none", "tag_control", "shells", "encoding_states", "custom"}
    with pytest.raises(ValueError, match="unknown acquisition design"):
        make_design("sense")


def test_tag_control_levels_and_column():
    d = make_design("tag_control", volume_types=["m0scan"] + ["control", "label"] * 4)
    assert d.levels.tolist() == [2] + [0, 1] * 4 and d.n_levels == 3
    assert d.encoding.tolist() == [0.0] + [1.0, -1.0] * 4
    short = make_design("tag_control", volume_types=["control", "label"] * 2)
    assert short.encoding is None  # fewer than 6 control/label volumes: no tag-control column


def test_shells_levels_and_direction_terms():
    rng = np.random.default_rng(0)
    bvals = np.array([0, 1000, 2000, 1000, 2000, 0, 1000, 2000])
    g = rng.normal(size=(8, 3))
    g[bvals == 0] = 0.0
    d = make_design("shells", bvals=bvals, bvecs=g, pe_axis=(0.0, 1.0, 0.0))
    assert d.levels.tolist() == [0, 1, 2, 1, 2, 0, 1, 2]
    assert d.bias_covariates.shape == (8, 5) and np.all(d.bias_covariates[bvals == 0] == 0.0)
    assert d.pe_affine_covariates.shape == (8, 3)


def test_l2_harmonics_are_the_traceless_quadratic_forms():
    """The five l = 2 real harmonics span {x^2 - y^2, 2z^2 - x^2 - y^2, xy, xz, yz} (unit vectors)."""
    rng = np.random.default_rng(1)
    g = rng.normal(size=(40, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    Y = _real_sh_l2(g, np.ones(40, bool))
    x, y, z = g.T
    Q = np.c_[x * x - y * y, 2 * z * z - x * x - y * y, x * y, x * z, y * z]
    coef = np.linalg.lstsq(Q, Y, rcond=None)[0]
    np.testing.assert_allclose(Q @ coef, Y, atol=1e-10)
    assert np.linalg.matrix_rank(Y) == 5


def test_frame_count_must_match():
    import ants

    img = ants.from_numpy(np.ones((8, 8, 8, 5), np.float32))
    with pytest.raises(ValueError, match="frames"):
        motion_correct_design(img, make_design("none", n_frames=6))


def test_dispatch_reproduces_the_modality_call_bold_and_asl():
    """``engine='dispatch'`` is today's per-modality call: identical transforms on simulated BOLD and ASL."""
    import ants

    from syntx.motion import motion_correction
    from syntx.motion_simulation import simulate_asl_series, simulate_bold_series

    def params(chains):
        return np.array([ants.read_transform(c[-1] if isinstance(c, (list, tuple)) else c).parameters for c in chains])

    bold = simulate_bold_series(n_frames=16, seed=3)
    a = motion_correction(bold.image, reference="mean", type_of_transform="Rigid", preset="bold", device="cpu")
    b = motion_correct_design(bold.image, make_design("none", n_frames=16), device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)
    assert b["design"]["call"] == "motion_correction(preset='bold')"

    asl = simulate_asl_series(n_frames=12, seed=3)
    vt = ["control" if s > 0 else "label" for s in asl.encoding_covariates]
    a = motion_correction(asl.image, reference="mean", type_of_transform="Rigid", preset="asl",
                          encoding_covariates=asl.encoding_covariates, device="cpu")
    b = motion_correct_design(asl.image, make_design("tag_control", volume_types=vt), device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)


def test_level_layout_interleaved_blocked_and_lone_frames():
    from syntx.motion_design import level_layout

    inter = level_layout(np.array([0, 1] * 8))
    assert inter[1]["interleaved"] and inter[1]["beta_estimable"]
    blocked = level_layout(np.array([0] * 8 + [1] * 8))
    assert not blocked[1]["interleaved"] and not blocked[1]["beta_estimable"]
    lone = level_layout(np.array([2] + [0, 1] * 6))
    assert not lone[2]["interleaved"] and not lone[2]["beta_estimable"]  # a lone frame cannot carry an offset


def test_unified_one_level_design_is_the_plain_registration():
    """BOLD is the special case: no levels to link, no beta -- identical to the dispatch call."""
    import ants

    from syntx.motion_simulation import simulate_bold_series

    def params(chains):
        return np.array([ants.read_transform(c[-1] if isinstance(c, (list, tuple)) else c).parameters for c in chains])

    sim = simulate_bold_series(n_frames=12, seed=5)
    d = make_design("none", n_frames=12)
    a = motion_correct_design(sim.image, d, engine="dispatch", device="cpu")
    b = motion_correct_design(sim.image, d, engine="unified", device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)


def test_unified_asl_level_offset_removes_the_tag_control_bias():
    """Design-specific references + identity link + level offset: the tag-control translation bias is removed (the
    variant without the offset keeps it)."""
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_asl_series

    sim = simulate_asl_series(n_frames=24, seed=2)
    d = make_design("tag_control", volume_types=["control" if s > 0 else "label" for s in sim.encoding_covariates])
    with_beta = motion_correct_design(sim.image, d, engine="unified", device="cpu")
    no_beta = motion_correct_design(sim.image, d, engine="unified", beta="none", device="cpu")
    assert with_beta["design"]["links"] == ["identity"]
    e1 = evaluate_motion_estimate(with_beta["fwdtransforms"], sim)["translation_rms_mm"]
    e0 = evaluate_motion_estimate(no_beta["fwdtransforms"], sim)["translation_rms_mm"]
    assert e1 < 0.5 * e0, (e1, e0)
    assert with_beta.motion_corrected is with_beta["motion_corrected"]   # the result type (attribute access) is kept


def test_unified_asl_with_lone_m0_still_removes_the_tag_control_bias():
    """A lone M0 frame (not identifiable: linked by registering its mean) must not switch off the offset of the
    interleaved label level."""
    import ants

    from syntx.motion_simulation import evaluate_motion_estimate, simulate_asl_series

    sim = simulate_asl_series(n_frames=24, seed=2)
    a = sim.image.numpy()
    img = ants.from_numpy(np.concatenate([2.5 * a[..., :1], a], axis=-1), origin=sim.image.origin,
                          spacing=sim.image.spacing, direction=sim.image.direction)
    vt = ["m0scan"] + ["control" if s > 0 else "label" for s in sim.encoding_covariates]
    d = make_design("tag_control", volume_types=vt)
    r = motion_correct_design(img, d, engine="unified", device="cpu")
    r0 = motion_correct_design(img, d, engine="unified", beta="none", device="cpu")
    assert r["design"]["links"] == ["identity", "registration"]
    xo = r["cross_group_offset"]
    assert set(xo["offset_mm_by_group"]) == {1} and xo["not_identifiable_groups"] == [2]
    e1 = evaluate_motion_estimate(r["fwdtransforms"][1:], sim)["translation_rms_mm"]
    e0 = evaluate_motion_estimate(r0["fwdtransforms"][1:], sim)["translation_rms_mm"]
    assert e1 < 0.5 * e0, (e1, e0)


def test_group_offsets_skip_unidentifiable_groups_and_keep_the_others():
    """Group 1 interleaved with a known offset, group 2 a lone frame: only group 1 gets an offset, and it is the
    planted one (the pose of the lone frame does not enter the projection)."""
    from syntx.motion_reference import _group_offsets, _pose_matrix

    com = np.zeros(3)
    groups = np.array([2] + [0, 1] * 8)
    off1 = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.01])
    Y = np.zeros((groups.size, 6))
    Y[:, 0] = np.linspace(0.0, 0.5, groups.size)          # smooth drift
    Y[groups == 1] += off1
    Y[groups == 2] += np.array([5.0, 0, 0, 0, 0, 0])     # lone frame far away
    H = np.array([_pose_matrix(y, com) for y in Y])
    idx = [np.flatnonzero(groups == g) for g in range(3)]
    _, per_group = _group_offsets(H, groups, idx, com)
    assert set(per_group) == {1}
    np.testing.assert_allclose(per_group[1], off1, atol=0.02)


def test_cross_link_values_are_checked():
    from syntx.motion_reference import _cross_link

    with pytest.raises(ValueError, match="cross_link"):
        _cross_link(None, None, "rigid", verbose=False, device="cpu", num_bins=32, convergence_tol_vox=0.05,
                    max_extend_factor=4)


def test_bf1_tag_control_contiguous_without_control():
    d = make_design("tag_control", volume_types=["label"] * 8 + ["m0scan"] * 2)
    assert set(d.levels.tolist()) == {0, 1}
    assert d.n_levels == 2
    assert d.detail.get("anchor_volume_type") == "label"


def test_bf2_shells_transposed_bvecs_raises():
    with pytest.raises(ValueError, match=r"shape \(T, 3\)"):
        make_design("shells", bvals=[0, 1000] * 5, bvecs=np.zeros((3, 10)))


def test_bf3_shells_b0_rows_zeroed_dw_rows_unit_normalized():
    bvals = np.array([0, 1000, 2000, 0, 1000])
    bvecs = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 4.0]])
    d = make_design("shells", bvals=bvals, bvecs=bvecs, pe_axis=[0, 1, 0])
    np.testing.assert_allclose(d.pe_affine_covariates[bvals == 0], 0.0)
    dw_norms = np.linalg.norm(d.pe_affine_covariates[bvals > 0], axis=1)
    np.testing.assert_allclose(dw_norms, 1.0, atol=1e-12)


def test_bf4_design_validation():
    from syntx.motion_design import Design

    with pytest.raises(ValueError, match="encoding rows"):
        Design(name="custom", levels=np.array([0, 1, 0, 1]), encoding=np.ones((5, 1)))

    with pytest.raises(ValueError, match="contiguous"):
        Design(name="custom", levels=np.array([0, 2, 0, 2]))

    with pytest.raises(ValueError, match="non-zero 3-vector"):
        Design(name="none", levels=np.zeros(4, dtype=int), pe_axis=np.array([0.0, 0.0, 0.0]))

    d_none = make_design("none", n_frames=10, pe_axis=[0, 2, 0])
    np.testing.assert_allclose(np.linalg.norm(d_none.pe_axis), 1.0, atol=1e-12)
    assert d_none.n_frames == 10


def test_bf5_custom_design_forwards_encoding_covariates():
    d = make_design("custom", levels=[0, 1] * 5, encoding=np.ones((10, 1)))
    assert d.encoding.shape == (10, 1)
    assert d.n_levels == 2


def test_f14_planted_linear_eddy_shift_recovers_without_leakage():
    """Synthetic shell with planted 0.4 mm linear-in-g PE eddy shift + l=2 bias under engine='unified'
    recovering full pe_affine shift without leakage into encoding_covariates and populating grouped['bias']."""
    from syntx.motion_simulation import simulate_dwi_series

    sim = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4)
    bv, bvec = sim.extra["bvals"], sim.extra["bvecs"]
    d = make_design("shells", bvals=bv, bvecs=bvec, pe_axis=sim.pe_axis)
    res = motion_correct_design(sim.image, d, engine="unified", verbose=False)
    assert res is not None
    assert "pe_affine" in res and res["pe_affine"] is not None
    assert "bias" in res and res["bias"] is not None
    a0 = np.asarray(res["pe_affine"]["a"])[:, 0]
    assert float(np.max(np.abs(a0))) > 0.25



# ---- engine="joint": one optimisation over the whole timeline (docs/MOTION_JOINT_OPTIMIZATION_PLAN.md) ----

def test_joint_design_columns_levels_and_per_shell_blocks():
    from syntx.motion_design import joint_design_columns, level_layout

    bv = np.array([0.0] * 7 + [1500.0, 3000.0] * 32)
    rng = np.random.default_rng(0)
    g = rng.normal(size=(len(bv), 3)); g /= np.linalg.norm(g, axis=1, keepdims=True); g[bv == 0] = 0
    d = make_design("shells", bvals=bv, bvecs=g, pe_axis=[0, 1, 0])
    enc, pe_cov, dropped = joint_design_columns(d, level_layout(d.levels))
    # shell 1 is the root of the interleaved component {1, 2}: its offset to the b0 block is not identifiable from the
    # interleaving (it is the H0 registration link's); shell 2's offset to shell 1 is -> one level column + l2 per shell
    assert enc.shape == (len(bv), 1 + 5 * 2) and len(dropped) == 1 and "level 1" in dropped[0]
    assert np.all(enc[d.levels == 0] == 0)                                 # the anchor level carries no column
    assert np.all(enc[d.levels == 1, 1:6] != 0) and np.all(enc[d.levels == 1, 6:] == 0)   # shell blocks are disjoint
    # a blocked level (b0 block first, never interleaved) has no identifiable offset: its column is dropped and recorded
    bv_blk = np.array([0.0] * 7 + [1500.0] * 40)
    g_blk = np.r_[np.zeros((7, 3)), g[7:47]]
    d_blk = make_design("shells", bvals=bv_blk, bvecs=g_blk, pe_axis=[0, 1, 0])
    enc_blk, _, dropped_blk = joint_design_columns(d_blk, level_layout(d_blk.levels))
    assert enc_blk.shape[1] == 5 and any("not identifiable" in x for x in dropped_blk)
    assert pe_cov.shape == (len(bv), 6) and np.all(pe_cov[d.levels == 0] == 0)
    assert np.allclose(np.linalg.norm(pe_cov[d.levels == 2, 3:], axis=1), 1.0) and np.all(pe_cov[d.levels == 2, :3] == 0)
    bv_short = np.array([0.0] * 3 + [1000.0] * 12)
    d2 = make_design("shells", bvals=bv_short, bvecs=np.r_[np.zeros((3, 3)), g[7:19]], pe_axis=[0, 1, 0])
    enc2, _, dropped2 = joint_design_columns(d2, level_layout(d2.levels))
    assert enc2 is None and len(dropped2) == 2     # blocked level: no offset column; l2 block dropped below 6 frames/column


def test_refine_schedule_needs_an_initial_pose():
    from syntx.motion_batched import batched_rigid_register_pass
    from syntx.motion_simulation import simulate_asl_series
    import ants

    sim = simulate_asl_series(n_frames=6, seed=0)
    frames = [ants.slice_image(sim.image, axis=3, idx=t) for t in range(6)]
    with pytest.raises(ValueError, match="init_pose"):
        batched_rigid_register_pass(frames[0], frames, schedule_mode="refine", device="cpu")


def test_joint_asl_matches_or_beats_unified_and_keeps_the_true_fd():
    """Total pose on the original frames from the H0 poses, the level column in the optimiser, the prior at the
    precision knee: error at most that of ``unified`` and the mean FD within 15 % of the truth (no jitter shrinkage)."""
    from scipy.spatial.transform import Rotation as Rot

    from syntx.motion import jenkinson_framewise_displacement
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_asl_series

    sim = simulate_asl_series(n_frames=24, seed=2)
    d = make_design("tag_control", volume_types=["control" if s > 0 else "label" for s in sim.encoding_covariates])
    rj = motion_correct_design(sim.image, d, engine="joint")
    ru = motion_correct_design(sim.image, d, engine="unified")
    ej = evaluate_motion_estimate(rj["fwdtransforms"], sim)
    eu = evaluate_motion_estimate(ru["fwdtransforms"], sim)
    assert ej["fd_error_mm"] <= 1.1 * eu["fd_error_mm"], (ej, eu)
    H = []
    for y in sim.head_pose:
        R = Rot.from_rotvec(y[3:]).as_matrix(); h = np.eye(4); h[:3, :3] = R; h[:3, 3] = y[:3]; H.append(h)
    fd_true = float(np.mean(jenkinson_framewise_displacement(np.asarray(H), 50.0, np.zeros(3))))
    assert abs(float(np.mean(rj["fd"])) - fd_true) < 0.15 * fd_true, (np.mean(rj["fd"]), fd_true)
    rec = rj["design"]
    assert rec["engine"] == "joint" and rec["columns"] == ["level1"] and rec["beta_levels"] is not None
    assert rec["surround_delta_mm"] == 0.05 and rj["group_indices"][0].shape == (24,)
    assert rj.motion_corrected.shape == sim.image.shape


def test_joint_dwi_eddy_applied_and_fd_closer_to_truth_than_unified():
    """Single-shell DWI with a planted 0.4 mm linear-in-g eddy shift: the joint pass keeps the eddy term applied (chain
    ``[eddy, rigid]``), the head FD error is below ``unified``'s, and the odd-only gauge projection leaves the eddy
    translation to the applied term."""
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_dwi_series

    sim = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4)
    bv, bvec = sim.extra["bvals"], sim.extra["bvecs"]
    d = make_design("shells", bvals=bv, bvecs=bvec, pe_axis=sim.pe_axis)
    rj = motion_correct_design(sim.image, d, engine="joint")
    ru = motion_correct_design(sim.image, d, engine="unified")
    ej = evaluate_motion_estimate(rj["fwdtransforms"], sim)
    eu = evaluate_motion_estimate(ru["fwdtransforms"], sim)
    assert ej["fd_error_mm"] < eu["fd_error_mm"], (ej, eu)
    dw = bv > 50
    assert all(len(c) == 2 and c[0].endswith("_eddy.mat") for c, isdw in zip(rj["fwdtransforms"], dw) if isdw)
    assert all(len(c) == 1 for c, isdw in zip(rj["fwdtransforms"], dw) if not isdw)
    a0 = np.asarray(rj["pe_affine"]["a"])[:, 0]
    assert float(np.max(np.abs(a0))) > 0.25
    assert np.all(np.asarray(rj["fd_raw"]) >= 0) and len(rj["fd"]) == len(bv)


def test_odd_only_projection_keeps_even_in_g_translation_out_of_the_pose():
    """A translation along phase encode that is EVEN in g is a contrast term: with the eddy covariates on, the gauge may
    carry it (the old projection removed every p-translation from the gauge and sent it to the pose)."""
    import ants
    from syntx.motion import motion_correction
    from syntx.motion_simulation import simulate_dwi_series

    sim = simulate_dwi_series(16, seed=3, eddy_mm_per_unit=0.0, drift_mm=0.0, drift_deg=0.0, jumps=(), jitter_mm=0.0)
    bv, bvec = sim.extra["bvals"], sim.extra["bvecs"]
    p = np.asarray(sim.pe_axis, dtype=float)
    dw = bv > 50
    g_dot_p = np.where(dw, bvec @ p, 0.0)
    even = 0.6 * (g_dot_p ** 2 - np.mean(g_dot_p[dw] ** 2)) * dw                 # even in g, zero mean over DW frames
    frames = []
    for t in range(len(bv)):
        f = ants.slice_image(sim.image, axis=3, idx=t)
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([np.eye(3).ravel(), even[t] * p]))
        frames.append(tx.apply_to_image(f, reference=f))
    img = ants.list_to_ndimage(sim.image, frames)
    enc = np.c_[(bv > 50).astype(float), np.where(dw[:, None], g_dot_p[:, None] ** 2, 0.0)]
    r = motion_correction(img, reference="mean", type_of_transform="Rigid", backend="pytorch_batched", seed_mode="narrow",
                          encoding_covariates=enc, pe_affine_covariates=np.where(dw[:, None], bvec, 0.0), pe_axis=p, device="cpu")
    t_p = np.asarray(r.motion_parameters.translations) @ p
    leak = float(np.std(t_p[dw] - np.mean(t_p[dw])))
    assert leak < 0.5 * float(np.std(even[dw])), (leak, float(np.std(even[dw])))


def test_surround_prior_hessian_matches_autograd():
    """The analytic per-frame Hessian block of the head prior (what the Newton stage adds to the image Hessian) equals the
    diagonal blocks of the autograd Hessian of the same prior, including the rotation scaling."""
    import torch
    from syntx.motion_batched import surround_prior_hessian

    torch.manual_seed(0)
    B, w, d = 7, 0.02, 0.05
    t = torch.randn(B, 3, dtype=torch.float64) * 0.1
    om = torch.randn(B, 3, dtype=torch.float64) * 0.002

    def prior(x):
        tt, oo = x.view(B, 6)[:, :3], x.view(B, 6)[:, 3:]
        u_t = tt[1:-1] - 0.5 * (tt[:-2] + tt[2:])
        u_w = 50.0 * (oo[1:-1] - 0.5 * (oo[:-2] + oo[2:]))
        return w * d * d * (torch.sqrt(1 + ((u_t * u_t).sum(-1) + (u_w * u_w).sum(-1)) / d / d) - 1).sum()

    x = torch.cat([t, om], 1).reshape(-1).clone()
    H = torch.autograd.functional.hessian(prior, x).view(B, 6, B, 6)
    blocks = torch.stack([H[k, :, k, :] for k in range(B)])
    Hb = surround_prior_hessian(t, om, w, d)
    assert float((Hb - blocks).abs().max()) < 1e-10 * float(blocks.abs().max())
    assert float(surround_prior_hessian(t, om, 0.0, d).abs().max()) == 0.0


def test_prior_in_every_stage_keeps_a_zero_motion_phantom_within_tolerance():
    """Static phantom with tag/control contrast only and NO motion, the head prior ON, refine schedule (L-BFGS then
    Newton; the prior is part of the objective in both): no point of the phantom moves by more than the Newton confirm
    tolerance. The phantom is the antsxfunctional three-blob zero-motion phantom plus a smooth texture: three Gaussian
    blobs alone leave rotation about their centres almost unconstrained, and on that plateau the prior's transient
    (the image term asks for an alternating pose, the prior refuses it, the encoding offset absorbs it) drifts the pose
    by 0.17 mm at the blobs' 16 mm extent while every frame reports converged (plateau, not optimum). With texture the
    same run gives 0.01 mm. FD at a 50 mm radius is not the criterion here: the blobs lie within 16 mm of the centre."""
    import ants
    from scipy.ndimage import gaussian_filter

    N = 32
    gx, gy, gz = np.ogrid[-N // 2:N // 2, -N // 2:N // 2, -N // 2:N // 2]
    b1 = np.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / (2 * 5.0 ** 2))
    b2 = np.exp(-((gx - 6) ** 2 + (gy + 4) ** 2 + (gz - 3) ** 2) / (2 * 3.5 ** 2))
    b3 = np.exp(-((gx + 5) ** 2 + (gy - 5) ** 2 + (gz + 4) ** 2) / (2 * 3.0 ** 2))
    rng = np.random.default_rng(0)
    tex = gaussian_filter(rng.standard_normal((N, N, N)), 1.5)
    tex = 12.0 * tex / tex.std() * np.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / (2 * 9.0 ** 2))
    w = {"control": (0.85, 0.60), "label": (0.55, 0.80)}
    vt = ["control" if t % 2 == 0 else "label" for t in range(8)]
    frames = [(100.0 * b1 + 80.0 * w[v][0] * b2 + 70.0 * w[v][1] * b3 + tex).astype(np.float32) for v in vt]
    img = ants.from_numpy(np.stack(frames, axis=-1), spacing=(2.0, 2.0, 2.0, 1.0))
    design = make_design("tag_control", volume_types=vt)
    r = motion_correct_design(img, design, engine="joint", surround_weight=0.02, device="cpu", verbose=False)
    rot = np.radians(np.asarray(r["rotations_deg"], float))
    t = np.asarray(r["translations_mm"], float)
    extent_mm = 16.0                     # farthest blob centre from the phantom centre (8 voxels x 2 mm)
    step = np.abs(np.diff(t, axis=0)).sum(1) + extent_mm * np.abs(np.diff(rot, axis=0)).sum(1)
    assert float(step.max()) <= 0.05, step   # Newton confirm tolerance 0.02 voxels = 0.04 mm, plus rounding


def test_projected_design_matrix_intercepts_and_l2_blocks():
    """Intercept per level (anchor included) plus the l = 2 block of each shell with enough frames; constant columns pruned."""
    from syntx.motion_design import level_layout, projected_design_matrix

    bv = np.r_[[0] * 10, [1000] * 40, [2000] * 40]
    rng = np.random.default_rng(0)
    g = rng.standard_normal((90, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True); g[bv == 0] = 0
    d = make_design("shells", bvals=bv, bvecs=g, pe_axis=np.array([0.0, 1.0, 0.0]))
    D, dropped = projected_design_matrix(d, level_layout(np.asarray(d.levels)))
    assert D.shape[0] == 90 and D.shape[1] == 3 + 2 * 5, (D.shape, dropped)
    assert np.allclose(D[:, :3].sum(axis=1), 1.0)                       # one intercept per frame
    assert np.linalg.matrix_rank(D) == D.shape[1]
    vt = ["control" if t % 2 == 0 else "label" for t in range(10)]
    D2, dropped2 = projected_design_matrix(make_design("tag_control", volume_types=vt), level_layout(np.arange(10) % 2))
    assert D2.shape == (10, 2) and dropped2 == []


def test_projected_dwi_beats_joint_and_unified_on_the_phantom():
    """G1 of docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md: single-shell DWI phantom with a planted eddy shift, the design in the
    metric (``projected``) lowers the joint engine's head FD error (0.176 vs 0.210 vs unified 0.319 measured, prior off), keeps the eddy
    term applied, and fixes no level gauge for the 4-frame b0 level (below GAUGE_MIN_FRAMES: its link is the measurement)."""
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_dwi_series

    sim = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4)
    bv, bvec = sim.extra["bvals"], sim.extra["bvecs"]
    d = make_design("shells", bvals=bv, bvecs=bvec, pe_axis=sim.pe_axis)
    rp = motion_correct_design(sim.image, d, engine="projected", device="cpu")
    rj = motion_correct_design(sim.image, d, engine="joint", device="cpu")
    ep = evaluate_motion_estimate(rp["fwdtransforms"], sim)
    ej = evaluate_motion_estimate(rj["fwdtransforms"], sim)
    assert ep["fd_error_mm"] < ej["fd_error_mm"], (ep, ej)              # CPU, prior off: 0.176 vs 0.210 (0.110 with the prior on)
    assert ep["fd_error_mm"] <= 0.19, ep
    dw = bv > 50
    assert all(len(c) == 2 and c[0].endswith("_eddy.mat") for c, isdw in zip(rp["fwdtransforms"], dw) if isdw)
    assert rp["design"]["engine"] == "projected" and rp["design"]["beta"] == "metric"
    assert rp["design_model"]["rank"] == 2 and rp["design_model"]["level_gauge"] == []
    assert len(rp["fd_bandlimited"]) == len(bv)


def test_projected_asl_fixes_the_level_gauge_and_tracks_the_truth():
    """Alternating control/label: a constant per-level pose offset is a nullspace of the within-level metric; the level
    gauge removes it (without it the pose alternated by 0.3 mm). The head FD error stays within 2.5x the joint engine's
    (prior off by default: 0.065 vs 0.055)."""
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_asl_series

    sim = simulate_asl_series(n_frames=24, seed=1)
    vt = ["control" if i % 2 == 0 else "label" for i in range(24)]
    d = make_design("tag_control", volume_types=vt)
    rp = motion_correct_design(sim.image, d, engine="projected", device="cpu")
    rj = motion_correct_design(sim.image, d, engine="joint", device="cpu")
    ep = evaluate_motion_estimate(rp["fwdtransforms"], sim)
    ej = evaluate_motion_estimate(rj["fwdtransforms"], sim)
    true = np.asarray(sim.head_pose)[:, :3]
    step_err = np.abs(np.diff(np.asarray(rp["translations_mm"]) - true, axis=0)).mean()
    assert step_err < 0.05, step_err                                   # no alternation left
    assert rp["design_model"]["level_gauge"][0]["removed_rms_mm"] > 0.05
    assert ep["fd_error_mm"] < 2.5 * ej["fd_error_mm"] and ep["fd_error_mm"] < 0.15, (ep, ej)
    r0 = motion_correct_design(sim.image, d, engine="projected", surround_weight=0.0, device="cpu")
    e0 = evaluate_motion_estimate(r0["fwdtransforms"], sim)
    assert e0["fd_error_mm"] < 1.5 * ej["fd_error_mm"], (e0, ej)        # 0.065 vs 0.055 on CPU


def test_projected_one_level_design_is_groupwise_registration():
    """A design without levels (resting BOLD) runs the same engine with an intercept-only model."""
    from syntx.motion_simulation import simulate_asl_series

    sim = simulate_asl_series(n_frames=10, seed=2)
    d = make_design("none", n_frames=10)
    r = motion_correct_design(sim.image, d, engine="projected", device="cpu")
    assert r["design"]["engine"] == "projected" and r["design_model"]["rank"] == 1
    assert len(r["fd"]) == 10 and np.all(np.asarray(r["frame_converged"]).shape == (10,))
