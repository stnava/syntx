"""Convergence-driven stopping of the linear solvers (syntx 6.17): the pose-step tracker, the per-frame converging stages of
motion_batched / robust_affine, and species-independence (the same voxels at 2 mm and at 0.2 mm give the same pose in voxels)."""
import numpy as np
import pytest
import torch
import ants
from scipy.ndimage import gaussian_filter

import syntx
from syntx.core.optimizers import PointStepTracker
from syntx.motion_batched import batched_rigid_register_pass


def _y(n_frames=3, n_pts=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_frames, n_pts, 3, generator=g)


class TestPointStepTracker:
    def test_converges_after_patience_checkpoints_of_small_steps(self):
        y0 = _y()
        t = PointStepTracker(tol=0.1, patience=2)
        t.start(y0)
        assert not t.update(y0 + 0.01).any()      # first small step: stable once
        assert t.update(y0 + 0.02).all()          # second: converged
        assert t.all_converged

    def test_large_step_resets_the_count(self):
        y0 = _y()
        t = PointStepTracker(tol=0.1, patience=2)
        t.start(y0)
        t.update(y0 + 0.01)
        t.update(y0 + 1.0)                        # moved a lot: not stable, count reset
        assert not t.update(y0 + 1.01).any()
        assert t.update(y0 + 1.02).all()

    def test_frames_are_independent(self):
        y0 = _y()
        t = PointStepTracker(tol=0.1, patience=2)
        t.start(y0)
        for k in range(1, 4):
            y = y0.clone()
            y[0] += 0.01 * k                      # frame 0 settles
            y[1] += 1.0 * k                       # frame 1 keeps moving
            y[2] += 0.01 * k
            t.update(y)
        assert t.active.tolist() == [False, True, False]
        assert t.last_step[1] > 0.5

    def test_step_limited_by_the_learning_rate_is_not_convergence(self):
        """A step that is small only because the optimiser's learning rate is small must not count (tail of a cosine schedule)."""
        y0 = _y(n_frames=1)
        t = PointStepTracker(tol=0.1, patience=2)
        t.start(y0)
        t.update(y0 + 0.05, capacity=0.06)        # moved 0.05 of a possible 0.06 -> lr-limited
        assert not t.update(y0 + 0.10, capacity=0.06).any()
        t.update(y0 + 0.101, capacity=5.0)        # free to move 5, moved ~0.001 -> converged evidence
        assert t.update(y0 + 0.102, capacity=5.0).all()

    def test_rejects_nonpositive_tol(self):
        with pytest.raises(ValueError):
            PointStepTracker(tol=0.0)


def _texture(n=36, seed=0):
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(size=(n, n, n)), 2.0)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    g = np.ogrid[-n // 2:n // 2, -n // 2:n // 2, -n // 2:n // 2]
    env = np.exp(-((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) / (2.0 * (0.3 * n) ** 2)))
    return (tex * env).astype(np.float32)


def _pair(spacing, shift_vox=(1.5, -1.0, 0.5), noise=0.01, seed=0):
    from scipy.ndimage import shift as ndshift
    base = _texture()
    rng = np.random.default_rng(seed)
    ref = base + rng.normal(0, noise, base.shape).astype(np.float32)
    mov = ndshift(base, shift_vox, order=3) + rng.normal(0, noise, base.shape).astype(np.float32)
    mk = lambda a: ants.from_numpy(np.clip(a, 0, None).astype(np.float32), spacing=spacing, origin=(0.0, 0.0, 0.0))
    return mk(ref), mk(mov)


def _t_vox(fwd_paths, spacing):
    from syntx.motion import _extract_rigid_parameters
    h = np.eye(4)
    for p in fwd_paths:
        h = _extract_rigid_parameters(ants.read_transform(p), 3)[2] @ h
    return h[:3, 3] / np.asarray(spacing)


class TestBatchedConvergence:
    def test_records_per_stage_convergence_and_reports_capped_stages(self, tmp_path):
        ref, mov = _pair((2.0, 2.0, 2.0))
        _, _, _, info = batched_rigid_register_pass(ref, [mov], verbose=False, outprefix=str(tmp_path / "a_"), seed_mode="narrow",
                                                    device="cpu", return_pe_affine=True)
        stages = info["convergence"]
        assert stages and all({"stage", "iterations", "blocks", "converged", "final_step_mm", "tol_mm"} <= set(s) for s in stages)
        assert all({"level"} <= set(s) for s in stages)
        assert all(all(s["converged"]) for s in stages if s["level"] == 1)           # the finest level certifies the result
        assert all(s["blocks"] == 1 for s in stages if s["level"] != 1)             # coarse levels only seed it: one block, never extended
        # an impossible tolerance with no extension: every stage must be REPORTED as capped, never silently final
        _, _, _, info2 = batched_rigid_register_pass(ref, [mov], verbose=False, outprefix=str(tmp_path / "b_"), seed_mode="narrow",
                                                     device="cpu", return_pe_affine=True, convergence_tol_vox=1e-9, max_extend_factor=1)
        assert not all(all(s["converged"]) for s in info2["convergence"] if s["level"] == 1)

    def test_same_voxels_at_mouse_and_human_spacing_give_the_same_pose_in_voxels(self, tmp_path):
        sp_h, sp_m = (2.0, 2.0, 2.0), (0.2, 0.2, 0.2)
        ref_h, mov_h = _pair(sp_h)
        ref_m, mov_m = _pair(sp_m)
        fwd_h, _, _ = batched_rigid_register_pass(ref_h, [mov_h], verbose=False, outprefix=str(tmp_path / "h_"), seed_mode="narrow", device="cpu")
        fwd_m, _, _ = batched_rigid_register_pass(ref_m, [mov_m], verbose=False, outprefix=str(tmp_path / "m_"), seed_mode="narrow", device="cpu")
        th, tm = _t_vox(fwd_h[0], sp_h), _t_vox(fwd_m[0], sp_m)
        assert np.max(np.abs(th - tm)) < 0.15, f"human {th} vs mouse {tm} voxels"
        assert np.allclose(th, [1.5, -1.0, 0.5], atol=0.35), th

    def test_frame_alone_and_in_a_batch_agree(self, tmp_path):
        ref, mov = _pair((2.0, 2.0, 2.0), seed=1)
        _, mov2 = _pair((2.0, 2.0, 2.0), shift_vox=(-1.0, 0.5, 1.0), seed=2)
        a, _, _ = batched_rigid_register_pass(ref, [mov], verbose=False, outprefix=str(tmp_path / "a_"), seed_mode="narrow", device="cpu")
        b, _, _ = batched_rigid_register_pass(ref, [mov2, mov, ref], verbose=False, outprefix=str(tmp_path / "b_"), seed_mode="narrow", device="cpu")
        assert np.max(np.abs(_t_vox(a[0], (2.0,) * 3) - _t_vox(b[1], (2.0,) * 3))) < 0.1
        assert np.max(np.abs(_t_vox(b[2], (2.0,) * 3))) < 0.1   # the reference itself stays at the identity


class TestRobustAffineConvergence:
    def test_reports_convergence_per_path_and_stage(self):
        ref, mov = _pair((2.0, 2.0, 2.0))
        res = syntx.robust_affine(ref, mov, mode="pytorch", dof="rigid", device="cpu")
        rec = res["convergence"]
        assert rec and all({"stage", "path", "iterations", "blocks", "converged", "final_step_mm"} <= set(r) for r in rec)
        assert rec[-1]["converged"]

    def test_same_voxels_at_mouse_and_human_spacing(self):
        sp_h, sp_m = (2.0, 2.0, 2.0), (0.2, 0.2, 0.2)
        out = []
        for sp in (sp_h, sp_m):
            ref, mov = _pair(sp)
            res = syntx.robust_affine(ref, mov, mode="pytorch", dof="rigid", device="cpu")
            out.append(_t_vox(res["fwdtransforms"], sp))
        assert np.max(np.abs(out[0] - out[1])) < 0.2, out


class TestMotionCorrectionReportsConvergence:
    def test_result_carries_per_frame_convergence(self):
        base = _texture(n=32)
        from scipy.ndimage import shift as ndshift
        frames = [base, ndshift(base, (0.8, -0.6, 0.5), order=3), ndshift(base, (-0.7, 0.9, -0.4), order=3)]
        img = ants.list_to_ndimage(ants.from_numpy(np.zeros(base.shape + (3,), np.float32), spacing=(2.0, 2.0, 2.0, 1.0)),
                                   [ants.from_numpy(f.astype(np.float32), spacing=(2.0, 2.0, 2.0)) for f in frames])
        res = syntx.motion_correction(img, reference=0, type_of_transform="Rigid", backend="pytorch_batched", verbose=False, device="cpu")
        conv = res["convergence"]
        assert conv is not None and conv["tol_vox"] == 0.05 and conv["max_extend_factor"] == 4
        assert len(conv["frame_converged"]) == 3 and all(conv["frame_converged"])
        assert conv["stages"]


from syntx.core.convergence import LossSlopeStop, loss_slope  # noqa: E402


class TestLossSlopeStop:
    def test_slope_is_per_iteration_whatever_the_sampling_cadence(self):
        its = [0, 5, 10, 15, 20]
        losses = [0.0, -0.05, -0.10, -0.15, -0.20]            # -0.01 per iteration, sampled every 5
        assert loss_slope(its, losses) == pytest.approx(-0.01)
        assert loss_slope(list(range(5)), losses) == pytest.approx(-0.05)

    def test_decreasing_or_diverging_loss_does_not_stop_a_flat_one_does(self):
        stop = LossSlopeStop(window=5, slope_threshold=1e-3)
        stop.start_level(0, cap=100)
        assert not any(stop.update(i, -0.01 * i) for i in range(20))      # still decreasing 0.01 / iteration
        stop.start_level(1, cap=100)
        hit = [stop.update(i, -1.0 + 1e-6 * i) for i in range(10)]
        assert hit[-1] and not hit[3]                                     # flat: converged once the window is full
        stop.start_level(2, cap=100)
        assert not any(stop.update(i, 0.01 * i) for i in range(12))       # rising fast = diverging: not convergence (caller's retry/best-field logic)

    def test_records_have_one_format_and_capped_levels_are_listed(self):
        stop = LossSlopeStop(window=4, slope_threshold=1e-3)
        stop.start_level(0, cap=50)
        for i in range(30):
            if stop.update(i, -0.01 * i):
                break
        stop.end_level(i + 1)
        stop.start_level(1, cap=30)
        for i in range(30):
            stop.update(i, 1.0)
            if stop.update(i, 1.0):
                break
        stop.end_level(i + 1)
        r0, r1 = stop.records
        assert set(r0) == {"level", "iterations", "cap", "converged", "final_slope"} == set(r1)
        assert not r0["converged"] and r0["iterations"] == 30 and r1["converged"] and r1["iterations"] < 30
        assert stop.summary()["capped_levels"] == [0] and not stop.all_converged

    def test_disabled_stop_never_converges_and_reports_capped(self):
        for thr in (None, 0.0):
            stop = LossSlopeStop(window=4, slope_threshold=thr)
            stop.start_level(0, cap=10)
            assert not any(stop.update(i, 1.0) for i in range(10))
            assert stop.end_level(10)["converged"] is False

    def test_rerun_of_a_level_keeps_the_last_attempt(self):
        stop = LossSlopeStop(window=4)
        stop.start_level(2, cap=10); stop.end_level(10)
        stop.start_level(2, cap=10); stop.end_level(7)
        assert len(stop.records) == 1 and stop.records[0]["iterations"] == 7

    def test_window_must_allow_a_slope(self):
        with pytest.raises(ValueError):
            LossSlopeStop(window=2)


def _blob2d(shift=(0.0, 0.0), n=32):
    g = np.ogrid[-n // 2:n // 2, -n // 2:n // 2]
    return ants.from_numpy(np.exp(-(((g[0] - shift[0]) ** 2 + (g[1] - shift[1]) ** 2) / (2.0 * 5.0 ** 2))).astype(np.float32))


def _check_record(conv, n_levels):
    assert conv is not None and len(conv["levels"]) == n_levels
    for r in conv["levels"]:
        assert {"level", "iterations", "cap", "converged", "final_slope"} <= set(r)
        assert 0 < r["iterations"] <= r["cap"]
    assert conv["capped_levels"] == [r["level"] for r in conv["levels"] if not r["converged"]]


class TestEveryDeformableMethodReportsTheSameRecord:
    def test_greedy(self):
        res = syntx.greedy_registration(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[40, 40], scales=[2, 1], device="cpu",
                                        initial_transform="identity", verbose=False)
        _check_record(res["convergence"], 2)
        loose = syntx.greedy_registration(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[100], scales=[1], device="cpu",
                                          initial_transform="identity", verbose=False, convergence_slope=1.0)
        assert loose["convergence"]["levels"][0]["converged"] and loose["convergence"]["levels"][0]["iterations"] < 100
        off = syntx.greedy_registration(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[30], scales=[1], device="cpu",
                                        initial_transform="identity", verbose=False, convergence_slope=None)
        assert off["convergence"]["levels"][0]["iterations"] == 30 and off["convergence"]["capped_levels"] == [0]

    def test_syn(self):
        res = syntx.registration(_blob2d(), _blob2d((1.5, -1.0)), type_of_transform="SyN", reg_iterations=[30, 20], device="cpu", verbose=False)
        _check_record(res["convergence"], 2)

    def test_tvf(self):
        res = syntx.tvf(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[30, 20], device="cpu", verbose=False)
        _check_record(res["convergence"], 2)

    def test_syngs(self):
        res = syntx.syngs(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[30, 20], device="cpu", verbose=False)
        _check_record(res["convergence"], 2)


class TestJaxBackendsReportTheSameRecord:
    def test_syn_jax(self):
        pytest.importorskip("jax")
        res = syntx.registration(_blob2d(), _blob2d((1.5, -1.0)), type_of_transform="SyN", reg_iterations=[30, 20], backend="jax", verbose=False)
        _check_record(res["convergence"], 2)

    def test_tvf_jax(self):
        pytest.importorskip("jax")
        res = syntx.tvf(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[30, 20], backend="jax", regularizer="gaussian", verbose=False)
        _check_record(res["convergence"], 2)

    def test_syngs_jax(self):
        pytest.importorskip("jax")
        res = syntx.syngs(_blob2d(), _blob2d((1.5, -1.0)), reg_iterations=[30, 20], backend="jax", optimizer="cfl", verbose=False)
        _check_record(res["convergence"], 2)
