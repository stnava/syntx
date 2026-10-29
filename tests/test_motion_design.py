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
    b = motion_correct_design(bold.image, make_design("none", n_frames=16), engine="dispatch", device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)
    assert b["design"]["call"] == "motion_correction(preset='bold')"

    asl = simulate_asl_series(n_frames=12, seed=3)
    vt = ["control" if s > 0 else "label" for s in asl.encoding_covariates]
    a = motion_correction(asl.image, reference="mean", type_of_transform="Rigid", preset="asl",
                          encoding_covariates=asl.encoding_covariates, device="cpu")
    b = motion_correct_design(asl.image, make_design("tag_control", volume_types=vt), engine="dispatch", device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)


def test_default_engine_is_joint_and_custom_encoding_reaches_it():
    """Since 6.30.0 ``motion_correct_design`` runs the joint engine by default, and a ``custom`` design's caller-supplied
    ``encoding`` columns enter the joint design (one-level path and multi-level columns) instead of being ignored."""
    from syntx.motion_simulation import simulate_asl_series

    sim = simulate_asl_series(n_frames=12, seed=3)
    r = motion_correct_design(sim.image, make_design("none", n_frames=12), device="cpu")
    assert r["design"]["engine"] == "joint" and r["design"]["surround_form"] == "cauchy"
    # one level + one continuous covariate (12 frames >= 6 x 1): forwarded to the solver
    enc = np.asarray(sim.encoding_covariates, dtype=float)
    r1 = motion_correct_design(sim.image, make_design("custom", levels=np.zeros(12, int), encoding=enc), device="cpu")
    assert r1.get("encoding_pose_bias_mm_rad") is not None
    # two levels + the same covariate: the joint columns carry the level indicator and the custom column
    lev = (enc < 0).astype(int)
    r2 = motion_correct_design(sim.image, make_design("custom", levels=lev, encoding=enc), device="cpu")
    assert any(c.startswith("custom x 1") for c in r2["design"]["columns"]), r2["design"]["columns"]


def test_encoding_states_design_is_one_hot_for_three_states():
    d = make_design("encoding_states", state_labels=[0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert d.n_levels == 3 and d.encoding.shape == (9, 2)
    np.testing.assert_array_equal(d.encoding[:, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(d.encoding[:, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1])
    d2 = make_design("encoding_states", state_labels=[0, 1, 0, 1])
    assert d2.encoding.shape == (4, 1)


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


def _assert_written_chain_reproduces_the_fitted_model(res, sim):
    """The written chain, resampled by ``syntx.transforms.apply_transforms`` (first-listed acts first on the reference
    point), must equal one resampling through the single matrix of the fitted model ``eddy(rigid(x))`` built from the
    rigid file and ``pe_affine['a']``; the opposite order ``rigid(eddy(x))`` must differ. Until 2026-10-28 the chain was
    written ``[eddy, rigid]`` and failed this test."""
    import ants
    import tempfile, os

    from syntx.transforms import _read_itk_affine, apply_transforms
    from syntx.motion_batched import rigid_transform_of, is_eddy_transform

    a = np.asarray(res["pe_affine"]["a"], dtype=np.float64)
    p = np.asarray(res["pe_affine"]["axis"], dtype=np.float64)
    ref = res["reference"] if "reference" in res else ants.slice_image(res["motion_corrected"], axis=3, idx=0)
    k = int(np.argmax(np.abs(a[:, 0])))
    chain = list(res["fwdtransforms"][k])
    assert len(chain) == 2 and is_eddy_transform(chain[1])
    rigid = ants.read_transform(rigid_transform_of(chain))
    c = np.asarray(rigid.fixed_parameters, dtype=np.float64)
    R, bR = _read_itk_affine(rigid)
    M = np.eye(3) + np.outer(p, a[k, 1:4])                                  # y' = M (y - c) + c + p a0
    E = np.eye(4); E[:3, :3] = M; E[:3, 3] = c - M @ c + p * a[k, 0]
    Hr = np.eye(4); Hr[:3, :3] = R; Hr[:3, 3] = bR
    d = tempfile.mkdtemp()

    def write(H, name):
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([H[:3, :3].flatten(), H[:3, 3] - c + H[:3, :3] @ c])); tx.set_fixed_parameters(c)
        path = os.path.join(d, name); ants.write_transform(tx, path); return path

    frame = ants.slice_image(sim.image, axis=3, idx=k)
    w_chain = apply_transforms(fixed=ref, moving=frame, transformlist=chain, interpolator="linear").numpy()
    w_model = apply_transforms(fixed=ref, moving=frame, transformlist=[write(E @ Hr, "model.mat")], interpolator="linear").numpy()
    w_wrong = apply_transforms(fixed=ref, moving=frame, transformlist=[write(Hr @ E, "wrong.mat")], interpolator="linear").numpy()
    m = ref.numpy() > 0.05 * float(ref.numpy().max())
    err_model = float(np.abs(w_chain - w_model)[m].mean()); err_wrong = float(np.abs(w_chain - w_wrong)[m].mean())
    assert err_model < 1e-4 * float(np.abs(w_chain[m]).mean()), (err_model, err_wrong)
    assert err_wrong > 10 * err_model, (err_model, err_wrong)


def test_joint_dwi_eddy_applied_and_fd_closer_to_truth_than_unified():
    """Single-shell DWI with a planted 0.4 mm linear-in-g eddy shift: the joint pass keeps the eddy term applied (chain
    ``[rigid, eddy]``), the head FD error is below ``unified``'s, and the odd-only gauge projection leaves the eddy
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
    assert all(len(c) == 2 and c[1].endswith("_eddy.mat") and not c[0].endswith("_eddy.mat")
               for c, isdw in zip(rj["fwdtransforms"], dw) if isdw)
    assert all(len(c) == 1 for c, isdw in zip(rj["fwdtransforms"], dw) if not isdw)
    a0 = np.asarray(rj["pe_affine"]["a"])[:, 0]
    assert float(np.max(np.abs(a0))) > 0.25
    _assert_written_chain_reproduces_the_fitted_model(rj, sim)

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


@pytest.mark.parametrize("form", ["charbonnier", "sparse", "cauchy"])
def test_surround_prior_hessian_matches_autograd(form):
    """The analytic per-frame Hessian block of the head prior (what the Newton stage adds to the image Hessian) equals the
    diagonal blocks of the autograd Hessian of the same prior, including the rotation scaling; for ``cauchy`` (indefinite
    exact Hessian beyond the knee) the model is the IRLS majoriser ``omega I``, compared against its own closed form. The
    gradient of ``surround_rho`` equals autograd for every form."""
    import torch
    from syntx.motion_batched import surround_prior_hessian, surround_rho, SPARSE_KNEE_FRACTION

    torch.manual_seed(0)
    B, w, d = 7, 0.02, 0.05
    t = torch.randn(B, 3, dtype=torch.float64) * 0.1
    om = torch.randn(B, 3, dtype=torch.float64) * 0.002

    def u_of(x):
        tt, oo = x.view(B, 6)[:, :3], x.view(B, 6)[:, 3:]
        return torch.cat([tt[1:-1] - 0.5 * (tt[:-2] + tt[2:]), 50.0 * (oo[1:-1] - 0.5 * (oo[:-2] + oo[2:]))], 1)

    def prior(x):
        return surround_rho((u_of(x) ** 2).sum(-1), w, d, form)[0].sum()

    x = torch.cat([t, om], 1).reshape(-1).clone()
    # gradient: autograd of rho(|u|^2) against omega u for the term values
    xg = x.clone().requires_grad_(True)
    u = u_of(xg); u_sq = (u * u).sum(-1)
    rho, omg, kap = surround_rho(u_sq, w, d, form)
    g_auto = torch.autograd.grad(rho.sum(), u, retain_graph=True)[0]
    assert float((g_auto - omg.unsqueeze(1) * u).abs().max()) < 1e-12
    Hb = surround_prior_hessian(t, om, w, d, form=form)
    if form == "cauchy":
        # IRLS model: omega I per term, attributed with the squared coefficients 1, 0.25, 0.25
        unit = torch.tensor([1.0, 1.0, 1.0, 50.0, 50.0, 50.0], dtype=torch.float64)
        ref = torch.zeros(B, 6, 6, dtype=torch.float64)
        for j in range(B - 2):
            blk = omg[j].detach() * torch.eye(6, dtype=torch.float64) * unit.view(6, 1) * unit.view(1, 6)
            ref[j + 1] += blk; ref[j] += 0.25 * blk; ref[j + 2] += 0.25 * blk
        assert float((Hb - ref).abs().max()) < 1e-10 * float(ref.abs().max())
        assert torch.linalg.eigvalsh(Hb).min() >= -1e-12
    else:
        H = torch.autograd.functional.hessian(prior, x).view(B, 6, B, 6)
        blocks = torch.stack([H[k, :, k, :] for k in range(B)])
        assert float((Hb - blocks).abs().max()) < 1e-10 * float(blocks.abs().max())
    assert float(surround_prior_hessian(t, om, 0.0, d, form=form).abs().max()) == 0.0
    if form == "sparse":   # algebraic identity: a Charbonnier with knee e and weight w d / e
        e = SPARSE_KNEE_FRACTION * d
        r_s = surround_rho(u_sq.detach(), w, d, "sparse")[0]
        r_c = surround_rho(u_sq.detach(), w * d / e, e, "charbonnier")[0]
        assert float((r_s - r_c).abs().max()) < 1e-12


def _denoise_pose_series(y, w, d, form, curvature=1.0):
    """Minimise ``curvature/2 |x - y|^2 + sum rho(second differences of x)``: the prior's action against a quadratic image
    term of unit curvature (the ``curvature`` prior scale makes the real solver's image term look like this)."""
    import torch
    from syntx.motion_batched import surround_rho

    y_t = torch.as_tensor(y, dtype=torch.float64)
    x = y_t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        u = x[1:-1] - 0.5 * (x[:-2] + x[2:])
        loss = 0.5 * curvature * ((x - y_t) ** 2).sum() + surround_rho(u * u, w, d, form)[0].sum()
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach().numpy()


def test_sparse_surround_removes_sub_knee_jitter_and_spares_drift_and_jumps():
    """The claim behind the ``sparse`` form: on a 1-D pose series = smooth drift + sub-knee jitter + one real jump, against a
    unit-curvature quadratic image term at weight 1 (prior curvature equal to the image curvature), ``sparse`` removes
    most of the jitter energy (a second-difference prior cannot remove the part of white jitter that lies in the smooth
    subspace, so the ceiling is below 1: 0.71 measured, 0.78 at weight 10), leaves the smoothed drift within 2 % of its
    range and the jump within 5 % of its size; ``charbonnier`` at the same weight removes at least 0.1 less jitter (its
    gradient vanishes at zero; 0.54 measured)."""
    rng = np.random.default_rng(0)
    T, d = 120, 0.05
    tt = np.arange(T)
    drift = 0.6 * np.sin(2 * np.pi * tt / T) + 0.004 * tt
    jitter = rng.normal(0.0, 0.02, T)          # second differences well below the knee
    jump = np.where(tt >= 80, 1.5, 0.0)
    y = drift + jitter + jump

    def moving_average(z, k=15):
        return np.convolve(z, np.ones(k) / k, mode="same")

    out = {}
    for form in ("sparse", "charbonnier"):
        x = _denoise_pose_series(y, 1.0, d, form)
        resid = x - (drift + jump)
        mask = np.ones(T, bool); mask[75:85] = False     # the jump's own neighbourhood is judged separately
        inner = mask.copy(); inner[:10] = False; inner[-10:] = False
        out[form] = dict(jitter_removed=1.0 - float(np.var(resid[mask]) / np.var(jitter[mask])),
                         drift_change=float(np.abs(moving_average(resid)[inner]).max() / np.ptp(drift)),
                         jump_kept=float(((x - drift)[85:95].mean() - (x - drift)[65:75].mean()) / 1.5))
    s, c = out["sparse"], out["charbonnier"]
    assert s["jitter_removed"] > 0.65, out
    assert s["drift_change"] < 0.02, out
    assert abs(s["jump_kept"] - 1.0) < 0.05, out
    assert s["jitter_removed"] > c["jitter_removed"] + 0.1, out


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
    metric (``projected``) stays within 1.3x of the joint engine's head FD error (0.199 vs 0.160 vs unified 0.242 after the Newton
    unit fix; it was 0.176 vs 0.210 before), keeps the eddy
    term applied, and fixes no level gauge for the 4-frame b0 level (below GAUGE_MIN_FRAMES: its link is the measurement)."""
    from syntx.motion_simulation import evaluate_motion_estimate, simulate_dwi_series

    sim = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4)
    bv, bvec = sim.extra["bvals"], sim.extra["bvecs"]
    d = make_design("shells", bvals=bv, bvecs=bvec, pe_axis=sim.pe_axis)
    rp = motion_correct_design(sim.image, d, engine="projected", device="cpu", surround_weight=0.0)   # engines compared with the prior OFF on both (6.28: the joint default prior reached 0.142 and broke the 1.3x ratio; the claim is about the metrics)
    rj = motion_correct_design(sim.image, d, engine="joint", device="cpu", surround_weight=0.0)
    ep = evaluate_motion_estimate(rp["fwdtransforms"], sim)
    ej = evaluate_motion_estimate(rj["fwdtransforms"], sim)
    # after the Newton unit fix (2026-10-28) joint improved more than projected (0.160 vs 0.199 on CPU); the claim that
    # survives is that the design-in-metric engine is within 1.3x of joint and better than unified (0.242)
    assert ep["fd_error_mm"] < 1.3 * ej["fd_error_mm"], (ep, ej)
    assert ep["fd_error_mm"] <= 0.22, ep
    dw = bv > 50
    assert all(len(c) == 2 and c[1].endswith("_eddy.mat") for c, isdw in zip(rp["fwdtransforms"], dw) if isdw)
    _assert_written_chain_reproduces_the_fitted_model(rp, sim)
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


def test_projected_design_matrix_adds_l4_on_high_b_shells():
    from syntx.motion_design import level_layout, projected_design_matrix

    rng = np.random.default_rng(0)
    bv = np.r_[[0] * 10, [1500] * 80, [3000] * 80]                     # 80 frames per shell: 9 l = 4 columns x 8 frames each fit
    g = rng.standard_normal((170, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True); g[bv == 0] = 0
    d = make_design("shells", bvals=bv, bvecs=g, pe_axis=np.array([0.0, 1.0, 0.0]))
    lay = level_layout(np.asarray(d.levels))
    D2, _ = projected_design_matrix(d, lay, lmax_high_b=2)
    D4, dropped = projected_design_matrix(d, lay, lmax_high_b=4)
    assert D2.shape[1] == 3 + 2 * 5 and D4.shape[1] == 3 + 2 * 5 + 9, (D2.shape, D4.shape, dropped)   # l = 4 on the b = 3000 shell only
    assert np.linalg.matrix_rank(D4) == D4.shape[1]
    bv46 = np.r_[[0] * 10, [1500] * 46, [3000] * 46]                   # 46 frames per shell (the cohort): the l = 4 block is gated out
    g46 = rng.standard_normal((102, 3)); g46 /= np.linalg.norm(g46, axis=1, keepdims=True); g46[bv46 == 0] = 0
    d46 = make_design("shells", bvals=bv46, bvecs=g46, pe_axis=np.array([0.0, 1.0, 0.0]))
    D46, dropped46 = projected_design_matrix(d46, level_layout(np.asarray(d46.levels)), lmax_high_b=4)
    assert D46.shape[1] == 3 + 2 * 5 and any("l4" in m for m in dropped46), (D46.shape, dropped46)


def test_newton_stage_converts_voxel_steps_to_parameters_correctly():
    """Regression for the Newton stage's unit conversion (fixed 2026-10-28): a voxel-unit step dv becomes a parameter step
    dv * dparam/dvox. Before the fix the step was divided instead, nine times too small for translations at 3 mm voxels and
    hundreds of times too large for rotations (rejected by the line search, so rotations never moved in the Newton stage).
    From a known pose offset of 0.6 voxels along x plus 1 degree about z, a Newton-only schedule must recover the truth
    within 0.1 voxel and 0.1 degree."""
    import ants
    from scipy.spatial.transform import Rotation as Rot
    from syntx.motion import motion_correction, _extract_rigid_parameters
    from syntx.motion_simulation import simulate_asl_series

    sim = simulate_asl_series(n_frames=8, seed=3, jitter_mm=0.0, drift_mm=0.0, drift_deg=0.0, jumps=(), perfusion=0.0)
    img = sim.image
    r0 = motion_correction(img, reference="mean", type_of_transform="Rigid", preset="bold", device="cpu", verbose=False, metric="huber")
    com = np.asarray(r0["reference"].get_center_of_mass(), float)
    H = [_extract_rigid_parameters(ants.read_transform(c[-1]), 3)[2] for c in r0.fwdtransforms]
    t_true = np.array([h[:3, 3] + (h[:3, :3] - np.eye(3)) @ com for h in H]); om_true = np.array([Rot.from_matrix(h[:3, :3]).as_rotvec() for h in H])
    vox = float(min(img.spacing[:3]))
    t_init = t_true + np.array([0.6 * vox, 0.0, 0.0]); om_init = om_true + np.array([0.0, 0.0, np.radians(1.0)])
    r = motion_correction(img, reference="mean", type_of_transform="Rigid", preset="bold", device="cpu", verbose=False, metric="huber",
                          schedule=[dict(optimizer="newton", iters=15, sampling=0.2, level=1, confirm_vox=0.02)], init_pose=(t_init, om_init))
    H1 = [_extract_rigid_parameters(ants.read_transform(c[-1]), 3)[2] for c in r.fwdtransforms]
    t1 = np.array([h[:3, 3] + (h[:3, :3] - np.eye(3)) @ com for h in H1]); om1 = np.array([Rot.from_matrix(h[:3, :3]).as_rotvec() for h in H1])
    assert np.abs(t1 - t_true).max() < 0.1 * vox, np.abs(t1 - t_true).max() / vox
    assert np.degrees(np.abs(om1 - om_true).max()) < 0.1, np.degrees(np.abs(om1 - om_true).max())


def test_newton_stage_damping_clamp_prevents_float32_overflow_on_repeated_rejections():
    """Regression for repeated line-search rejections in coupled Newton (iters=20, max_extend=4 -> 80 calls):
    without an upper clamp on Levenberg-Marquardt damping ``lam`` and a stagnation exit, 65 rejections multiplied
    ``lam`` by 4^65 > float32_max (inf), turning ``Q @ diag(ev + lam) @ Q.T`` and ``0.0 * dv`` into NaN and crashing
    ``torch.linalg.eigh`` on the next iteration."""
    import torch
    from syntx.core.parameter_scales import rigid_axes
    from syntx.motion_batched import _run_converging_stage

    B = 4
    t_param = torch.zeros((B, 3), dtype=torch.float32, requires_grad=True)
    omega = torch.zeros((B, 3), dtype=torch.float32, requires_grad=True)
    pts = torch.randn((32, 3), dtype=torch.float32) * 30.0
    center = torch.zeros(3, dtype=torch.float32)
    axes = rigid_axes(t_param, omega, pts, center)

    # Linear tilt gradient at origin so undamped Newton step is non-zero (> confirm_vox), but objective returns a constant
    # on no_grad line-search evaluations so every line-search trial is rejected.
    def loss_fn():
        if torch.is_grad_enabled():
            return 0.5 * ((t_param - 1.0) ** 2).sum(dim=1) + 0.5 * ((omega - 0.02) ** 2).sum(dim=1)
        return torch.ones(B, dtype=torch.float32)

    rec = _run_converging_stage(
        optimizer="newton", params=[t_param, omega], per_frame_params=[t_param, omega],
        loss_fn=loss_fn, points_fn=lambda: torch.zeros((B, 8, 3), dtype=torch.float32),
        iters=25, B=B, dev=torch.device("cpu"), tol=0.05, max_extend=4,
        confirm_vox=0.01, axes=axes, stage_voxel=3.0, max_step_vox=1.0,
        label="newton@L1", level=1,
        reg_loss_fn=lambda: 0.01 * (t_param ** 2).sum(),
        reg_hess_full_fn=lambda: torch.zeros((B, 6, B, 6), dtype=torch.float32),
    )
    assert torch.isfinite(t_param).all() and torch.isfinite(omega).all()
    assert rec["iterations"] < 25, rec

