"""robust_affine(setting='intermodality_slab'): translation-first, multi-start, native-MI-ranked thin-slab capture (syntx.slab_capture).

The strategy tests use a stub single-start solver (records its calls, returns prescribed poses) on a real synthetic slab pair, so
they pin the logic (what is called, in what order, what is ranked by what) without optimiser run time. The end-to-end tests use
the real solver. Evidence and measured tables: docs/RELEASE_NOTES_6.3.0.md.
"""
from __future__ import annotations

import importlib
import os
import tempfile

import ants
import numpy as np
import pytest
import torch

import syntx
from syntx import RobustAffineSolverError
from syntx import slab_capture as SC

from tests.test_robust_affine_anisotropy import _phantom, _slab_pair, _tre

RA = importlib.import_module("syntx.robust_affine")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------------------------------------------ helpers
def _tx_file(A, t_eff):
    return SC._write_affine(np.asarray(A, dtype=float), np.asarray(t_eff, dtype=float), os.path.join(tempfile.mkdtemp(), "t.mat"))


def _truth(tx):
    p = np.asarray(tx.parameters, dtype=float)
    return p[:9].reshape(3, 3), p[9:12]          # fixed parameters are zero in _slab_pair: y = A x + t


class StubSolver:
    """Records every call; ``plan(call_index, kwargs) -> (A, t_eff, coarse_loss)`` or raises."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, fixed, moving, **kw):
        i = len(self.calls)
        self.calls.append(kw)
        A, t, coarse = self.plan(i, kw)
        path = _tx_file(A, t)
        return {"fwdtransforms": [path], "self_check": {"final_loss_coarse": coarse}, "init_candidate": "stub", "final_loss": coarse}


@pytest.fixture(scope="module")
def slab():
    return _slab_pair(4)


# ------------------------------------------------------------------------------------------------------ validation
@pytest.mark.parametrize("bad", ["slab", "", "Default", None, 1, ["default"]])
def test_setting_value_is_validated(bad):
    img = ants.from_numpy(np.random.default_rng(0).random((16, 16, 16)).astype(np.float32))
    with pytest.raises(ValueError, match="setting"):
        syntx.robust_affine(img, img, mode="auto", device="cpu", setting=bad)


@pytest.mark.parametrize("kw,match", [
    (dict(dof="affine"), "dof='rigid'"),
    (dict(dof="rigid", mode="ants"), "mode must be"),
    (dict(dof="rigid", tournament=True), "mode must be"),
    (dict(dof="rigid", levels=[4, 2, 1]), "levels / iterations / schedule"),
    (dict(dof="rigid", schedule=[dict(level=1)]), "levels / iterations / schedule"),
])
def test_slab_setting_rejects_unsupported_combinations(kw, match):
    img = ants.from_numpy(np.random.default_rng(0).random((16, 16, 16)).astype(np.float32))
    with pytest.raises(ValueError, match=match):
        syntx.robust_affine(img, img, device="cpu", setting="intermodality_slab", **kw)


def test_strategy_arguments_are_validated(slab):
    fixed, mov, _ = slab
    for kw in (dict(n_starts=0), dict(n_starts=2.5), dict(max_rotation_deg=-1.0), dict(max_rotation_deg=400.0)):
        with pytest.raises(ValueError):
            SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, solver=StubSolver(lambda i, k: (np.eye(3), np.zeros(3), 0.0)), **kw)


def test_default_setting_is_the_default_path():
    """setting='default' is the unchanged solver: same transform as not passing the argument."""
    iso = _phantom()
    mov = ants.apply_transforms(fixed=iso, moving=iso, transformlist=[], interpolator="linear")
    a = syntx.robust_affine(iso, mov, mode="pytorch", dof="rigid", device="cpu", preset="fast", seed=3)
    b = syntx.robust_affine(iso, mov, mode="pytorch", dof="rigid", device="cpu", preset="fast", seed=3, setting="default")
    np.testing.assert_array_equal(ants.read_transform(a["fwdtransforms"][0]).parameters, ants.read_transform(b["fwdtransforms"][0]).parameters)
    assert "slab_capture" not in b


# ------------------------------------------------------------------------------------------------------ strategy
def test_translation_stage_is_translation_only_and_runs_first(slab):
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)
    stub = StubSolver(lambda i, k: (np.eye(3), t_t, -0.3))
    SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=3, solver=stub)
    first = stub.calls[0]
    assert first["initial_tx_path"] is None and first["cone_angles_deg"] == []        # no rotation candidates: translation only
    assert all(s["lr"][1] == 0.0 and s["eta_min"] == 0.0 for s in first["schedule"])  # the rotation group never moves
    assert all(s["dof"] == "rigid" for s in first["schedule"])
    assert len(stub.calls) == 1 + 3                                                    # translation + n_starts refinements
    assert all(c["multi_start"] is False and c["initial_tx_path"] is not None for c in stub.calls[1:])


def test_multistart_rotations_are_random_bounded_and_composed_on_the_translation(slab):
    fixed, mov, _ = slab
    t_tr = np.array([2.0, -3.0, 11.0])
    stub = StubSolver(lambda i, k: (np.eye(3), t_tr, -0.2) if i == 0 else (np.eye(3), t_tr, -0.2))
    n, max_deg = 9, 30.0
    SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=n, max_rotation_deg=max_deg, seed=5, solver=stub)
    C = SC._fov_center(fixed)
    Rs = []
    for c in stub.calls[1:]:
        A, t_eff = SC._affine_from_file(c["initial_tx_path"])
        assert np.allclose(A @ A.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(A), 1.0, atol=1e-6)
        np.testing.assert_allclose(t_eff, C - A @ C + t_tr, atol=1e-4)                  # y = R (x - C) + C + t_translation
        Rs.append(A)
    assert np.allclose(Rs[0], np.eye(3), atol=1e-6)                                      # start 0: the translation alone
    angles = [np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) for R in Rs[1:]]
    assert min(angles) > 0.5 and max(angles) <= np.sqrt(3) * max_deg + 1e-3
    assert len({tuple(np.round(R.ravel(), 4)) for R in Rs}) == n                         # all distinct
    # deterministic in the seed
    stub2 = StubSolver(lambda i, k: (np.eye(3), t_tr, -0.2))
    SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=n, max_rotation_deg=max_deg, seed=5, solver=stub2)
    for c1, c2 in zip(stub.calls[1:], stub2.calls[1:]):
        np.testing.assert_allclose(SC._affine_from_file(c1["initial_tx_path"])[0], SC._affine_from_file(c2["initial_tx_path"])[0], atol=1e-6)


def test_native_mi_ranking_beats_coarse_loss_ranking(slab):
    """Constructed case: the wrong pose (7 mm off) has the BETTER coarse-level loss, the true pose the better native MI. Native MI wins."""
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)
    wrong_t = t_t + np.array([0.0, 0.0, 7.0])
    poses = {1: (A_t, t_t, -0.10), 2: (A_t, wrong_t, -0.40)}              # call 0 is the translation stage; coarse loss: wrong looks better
    stub = StubSolver(lambda i, k: poses[i] if i in poses else (np.eye(3), t_t, -0.35))
    reg = SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=2, solver=stub)
    info = reg["slab_capture"]
    assert info["coarse_loss_ranking"][0][1] == "start01"                   # the solver's own ranking would pick the wrong pose
    assert info["winner"] == "start00_translation"
    assert _tre(reg, tx) < 0.5
    assert dict(info["ranking"])["start00_translation"] > dict(info["ranking"])["start01"]


def test_default_solver_result_joins_the_pool_and_can_win(slab):
    """The ordinary solver's pose is one more member of the pool ranked by native MI (every translation-first start can share one wrong basin)."""
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)
    wrong_t = t_t + np.array([0.0, 0.0, 7.0])

    def plan(i, kw):
        if i == 0:
            return np.eye(3), wrong_t, -0.3                     # translation stage in the wrong basin
        return (A_t, t_t, -0.2) if kw["multi_start"] else (A_t, wrong_t, -0.3)   # the multi-start default call finds the truth
    stub = StubSolver(plan)
    reg = SC.run_slab_capture(fixed, mov, device="cpu", n_starts=2, solver=stub)
    assert reg["slab_capture"]["winner"] == "default_solver" and len(stub.calls) == 1 + 2 + 1
    assert stub.calls[-1]["multi_start"] is True and stub.calls[-1]["n_starts"] == 3
    assert _tre(reg, tx) < 0.5


def test_z_shift_strategy_shifts_only_along_the_thick_axis(slab):
    """strategy='z_shift' (a documented negative result): no translation stage, the default solver plus eight identity-rotation starts whose
    translations differ from the centre-of-mass start by the prescribed offsets along the thick axis only."""
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)
    stub = StubSolver(lambda i, k: (A_t, t_t, -0.2))
    reg = SC.run_slab_capture(fixed, mov, device="cpu", strategy="z_shift", solver=stub)
    assert len(stub.calls) == 8 + 1 and stub.calls[-1]["multi_start"] is True           # 8 shifted starts + the default solver
    t0 = SC._com_translation(fixed, mov)
    ax = int(np.argmax(fixed.spacing))
    zdir = np.asarray(fixed.direction).reshape(3, 3)[:, ax]
    for c, off in zip(stub.calls[:8], SC.Z_OFFSETS_MM):
        A, t = SC._affine_from_file(c["initial_tx_path"])
        np.testing.assert_allclose(A, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(t, t0 + off * zdir, atol=1e-3)
        assert c["multi_start"] is False
    assert reg["slab_capture"]["n_starts"] == 8
    with pytest.raises(ValueError, match="strategy"):
        SC.run_slab_capture(fixed, mov, device="cpu", strategy="x", solver=stub)


def test_blank_warp_is_rejected(slab):
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)
    far = t_t + np.array([0.0, 0.0, 5000.0])                               # the moving image leaves the field of view: zero warp
    stub = StubSolver(lambda i, k: (np.eye(3), t_t, -0.2) if i == 0 else ((A_t, t_t, -0.2) if i == 2 else (A_t, far, -0.9)))   # the blank pose even has the best "loss"
    reg = SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=2, solver=stub)
    assert reg["slab_capture"]["winner"] == "start01"
    assert ("start00_translation", "blank warped image") in reg["slab_capture"]["rejected"]
    stub = StubSolver(lambda i, k: (np.eye(3), t_t, -0.2) if i == 0 else (A_t, far, -0.9))
    with pytest.raises(RobustAffineSolverError, match="none of the"):
        SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=2, solver=stub)


def test_failing_starts_are_recorded_and_all_failing_raises(slab):
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)

    def plan(i, kw):
        if i in (1, 2):
            raise RobustAffineSolverError("synthetic start failure")
        return (np.eye(3) if i == 0 else A_t), t_t, -0.2
    reg = SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=3, solver=StubSolver(plan))
    assert [n for n, why in reg["slab_capture"]["rejected"]] == ["start00_translation", "start01"]
    assert reg["slab_capture"]["winner"] == "start02"

    def plan_all(i, kw):
        if i == 0:
            return np.eye(3), t_t, -0.2
        raise RobustAffineSolverError("synthetic start failure")
    with pytest.raises(RobustAffineSolverError, match="none of the"):
        SC.run_slab_capture(fixed, mov, device="cpu", include_default_start=False, n_starts=2, solver=StubSolver(plan_all))


def test_native_scorer_prefers_truth_and_applies_the_valid_margin(slab):
    fixed, mov, tx = slab
    A_t, t_t = _truth(tx)
    s1 = SC.NativeMIScorer(fixed, mov, torch.device("cpu"), valid_margin_slices=1)
    l_true, blank = s1.loss(A_t, t_t)
    assert not blank and l_true < s1.loss(A_t, t_t + np.array([0, 0, 4.0]))[0] and l_true < s1.loss(A_t, t_t + np.array([4.0, 0, 0]))[0]
    assert s1.valid.sum() < s1.valid.numel()                                   # the slab-edge slices are not in the metric
    s0 = SC.NativeMIScorer(fixed, mov, torch.device("cpu"), valid_margin_slices=0)
    assert s0.valid.all() and s0.loss(A_t, t_t)[0] != l_true
    assert s1.loss(np.eye(3), np.array([0.0, 0.0, 9000.0]))[1] is True       # blank


# ------------------------------------------------------------------------------------------------------ end to end (real solver)
def test_translation_stage_captures_a_large_offset():
    """The T1 is cropped at one end so its centre of mass (the start of every default candidate) is several millimetres off along z;
    the translation-first stage must bring the pose to < 1.5 mm of the known rigid transform."""
    fixed, mov, tx = _slab_pair(4)
    n = mov.shape[2]
    moving = ants.crop_indices(mov, (0, 0, 14), (mov.shape[0], mov.shape[1], n))
    reg = syntx.robust_affine(fixed, moving, mode="auto", dof="rigid", device=DEV, seed=42, setting="intermodality_slab")
    assert reg["solver"] == "pytorch" and reg["setting"] == "intermodality_slab"
    assert _tre(reg, tx) < 1.5
    assert abs(np.linalg.norm(reg["slab_capture"]["translation_mm"])) > 0


def test_thin_slab_6to1_pose_a_is_solved_by_the_intermodality_slab_setting():
    """The one default-path synthetic failure (strict xfail in test_robust_affine_anisotropy.py: 4.4 mm): the capture setting solves it."""
    fixed, mov, tx = _slab_pair(6)
    reg = syntx.robust_affine(fixed, mov, mode="auto", dof="rigid", device=DEV, seed=42, setting="intermodality_slab")
    assert _tre(reg, tx) < 1.5


_D = "/Users/stnava/data/nm_registration_debug/subjects"


@pytest.mark.slow
def test_real_nm_t1_all_ten_pairs_within_2mm_of_ants_with_the_slab_setting():
    """Ten real ds001242 neuromelanin-slab / T1 pairs (local data; slow: 32 starts per pair): every pose within 2 mm of the ANTs rigid reference."""
    subs = [f"sub-{i:02d}" for i in range(1, 11)]
    need = ("fixed_nm_n4.nii.gz", "moving_t1_slab_crop.nii.gz", "ants_rigid_reference.mat")
    if not all(os.path.exists(f"{_D}/{s}/{f}") for s in subs for f in need):
        pytest.skip("local NM debug data absent")
    errs = {}
    for sub in subs:
        d = f"{_D}/{sub}"
        fixed = ants.image_read(f"{d}/fixed_nm_n4.nii.gz"); moving = ants.image_read(f"{d}/moving_t1_slab_crop.nii.gz")
        c = np.array(ants.get_center_of_mass(fixed * 0 + 1))
        probes = [c + np.array([x, y, z]) for x in (-15, 15) for y in (-15, 15) for z in (-5, 5)] + [c]

        def disp(p):
            t = ants.read_transform(p)
            return np.array([np.array(t.apply_to_point(tuple(q))) - q for q in probes])

        reg = syntx.robust_affine(fixed, moving, mode="auto", dof="rigid", seed=42, device=DEV, setting="intermodality_slab")
        errs[sub] = float(np.linalg.norm(disp(reg["fwdtransforms"][0]) - disp(f"{d}/ants_rigid_reference.mat"), axis=1).max())
    assert max(errs.values()) < 2.0, errs
