"""
Benchmark provenance (syntx.provenance): results must identify the exact code, script,
environment and resolved parameters that produced them.
"""

import copy
import json
import os
import subprocess

import numpy as np
import pytest
import ants

import syntx
from syntx.provenance import (
    assert_manifest_complete,
    build_manifest,
    capture_registration_calls,
    code_state,
    cohort_provenance,
    resolved_parameters,
    with_provenance,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pair(n=24):
    arr = np.random.default_rng(0).random((n, n, n)).astype(np.float32)
    return ants.from_numpy(arr), ants.from_numpy(np.roll(arr, 1, 0))


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


# ----------------------------------------------------------------------------------------
# code state
# ----------------------------------------------------------------------------------------
def test_code_state_is_the_imported_checkout():
    st = code_state()
    assert st["syntx_version"] == syntx.__version__
    assert st["git"]["commit"] == _git(ROOT, "rev-parse", "HEAD")
    assert len(st["git"]["diff_sha256"]) == 64


def test_code_state_records_tracked_diff_and_untracked_package_sources(tmp_path):
    repo = tmp_path / "repo"
    pkg = repo / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")

    clean = code_state(pkg_dir=str(pkg))["git"]
    assert clean["dirty"] is False and clean["diff"] == ""

    (pkg / "a.py").write_text("x = 2\n")                       # tracked change
    (pkg / "new_module.py").write_text("y = 3\n")              # untracked package source
    (repo / "notes.txt").write_text("not code\n")              # untracked, outside package
    dirty = code_state(pkg_dir=str(pkg))["git"]
    assert dirty["dirty"] is True
    assert "+x = 2" in dirty["diff"]
    assert dirty["untracked_package_files"] == ["src/pkg/new_module.py"]
    assert dirty["untracked_package_sources"]["src/pkg/new_module.py"] == "y = 3\n"
    assert "notes.txt" in dirty["untracked"]
    assert dirty["diff_sha256"] != clean["diff_sha256"]


# ----------------------------------------------------------------------------------------
# call capture
# ----------------------------------------------------------------------------------------
def test_capture_records_explicit_and_resolved_syn_parameters():
    f, m = _pair()
    orig_syn, orig_fit = syntx.syn, syntx.SyNTo.fit
    with capture_registration_calls() as cap:
        syntx.syn(fixed=f, moving=m, initial_transform="identity", reg_iterations=[2, 1, 1],
                  device="cpu", fast_smooth=True)
    assert syntx.syn is orig_syn and syntx.SyNTo.fit is orig_fit  # restored

    man = build_manifest(calls=cap.calls, run={"test": True})
    assert_manifest_complete(man)
    json.dumps(man)  # serialisable
    p = resolved_parameters(man)
    assert p["function"] == "syntx.syn"
    assert p["explicit"]["fast_smooth"] is True
    # hidden / body-resolved defaults are visible
    assert p["fit_kwargs"]["sobolev_alpha"] == 1.5
    assert p["fit_kwargs"]["cfl_voxels"] == 0.25
    assert p["model_attributes"]["in_loop_inv_steps"] == 10
    assert p["model_attributes"]["stationary_boundary"] is True


def test_capture_records_ants_registration_and_restores_on_error():
    f, m = _pair(16)
    orig = ants.registration
    with pytest.raises(Exception):
        with capture_registration_calls() as cap:
            ants.registration(fixed=f, moving=m, type_of_transform="Translation")
            syntx.syn(fixed=None, moving=m)
    assert ants.registration is orig
    fns = [(c["function"], c["status"]) for c in cap.calls]
    assert fns[0] == ("ants.registration", "ok")
    assert fns[1][0] == "syntx.syn" and fns[1][1].startswith("error")
    assert cap.calls[0]["kwargs"]["type_of_transform"] == "Translation"


def test_with_provenance_decorator_attaches_manifest():
    f, m = _pair(16)

    @with_provenance("tests.dummy_evaluator")
    def evaluator(fixed, moving, step=0.25):
        syntx.syn(fixed=fixed, moving=moving, initial_transform="identity",
                  reg_iterations=[1, 1, 1], grad_step=step, device="cpu")
        return {"dice": 0.5}

    out = evaluator(f, m)
    man = out["provenance"]
    assert_manifest_complete(man)
    assert man["run"]["evaluator"] == "tests.dummy_evaluator"
    assert man["run"]["step"] == 0.25  # defaults applied
    assert man["script"]["argv"]


def test_manifest_without_commit_or_calls_is_rejected():
    man = build_manifest(calls=[], run={})
    with pytest.raises(ValueError, match="no registration call"):
        assert_manifest_complete(man)
    bad = copy.deepcopy(man)
    bad["code"]["git"] = None
    with pytest.raises(ValueError, match="git commit"):
        assert_manifest_complete(bad, require_calls=False)


# ----------------------------------------------------------------------------------------
# cohort summaries
# ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def one_manifest():
    f, m = _pair(16)
    with capture_registration_calls() as cap:
        syntx.syn(fixed=f, moving=m, initial_transform="identity", reg_iterations=[1, 1, 1],
                  device="cpu")
    return build_manifest(calls=cap.calls, run={}, include_diff=False)


def test_cohort_provenance_consistent(one_manifest):
    summary = cohort_provenance([one_manifest, copy.deepcopy(one_manifest)])
    assert summary["n_runs"] == 2
    assert summary["commit"] == one_manifest["code"]["git"]["commit"]
    assert summary["parameters"]["fit_kwargs"]["sobolev_alpha"] == 1.5


def test_cohort_provenance_rejects_mixed_code_or_parameters(one_manifest):
    other_code = copy.deepcopy(one_manifest)
    other_code["code"]["git"]["diff_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="code states"):
        cohort_provenance([one_manifest, other_code])

    other_params = copy.deepcopy(one_manifest)
    other_params["calls"][0]["resolved"][-1]["fit_kwargs"]["cfl_voxels"] = 0.35
    with pytest.raises(ValueError, match="parameter sets"):
        cohort_provenance([one_manifest, other_params])


def test_manifest_uses_start_state_and_flags_changes_during_run(monkeypatch):
    import syntx.provenance as prov
    f, m = _pair(16)
    with capture_registration_calls() as cap:
        syntx.syn(fixed=f, moving=m, initial_transform="identity", reg_iterations=[1, 1, 1],
                  device="cpu")
    start = cap.start_state
    unchanged = build_manifest(calls=cap.calls, run={}, start=start)
    assert unchanged["changed_during_run"] is False
    assert unchanged["code"] is start["code"]

    # simulate an edit on disk while the run executed
    real = prov.code_state

    def edited(*a, **k):
        st = real(*a, **k)
        st["git"]["diff_sha256"] = "f" * 64
        return st

    monkeypatch.setattr(prov, "code_state", edited)
    changed = build_manifest(calls=cap.calls, run={}, start=start)
    assert changed["changed_during_run"] is True
    assert changed["code"]["git"]["diff_sha256"] == start["code"]["git"]["diff_sha256"]
    assert changed["at_end"]["code"]["diff_sha256"] == "f" * 64
    with pytest.raises(ValueError, match="changed on disk"):
        cohort_provenance([changed])


def test_cohort_provenance_ignores_per_image_fit_inputs(one_manifest):
    """Runs on different pairs share parameters but not image tensors / initial affines."""
    other = copy.deepcopy(one_manifest)
    fk = other["calls"][0]["resolved"][-1]["fit_kwargs"]
    fk["fixed_tensor"] = "<tensor shape=(1, 1, 99, 99, 99) dtype=torch.float32 device=cpu>"
    fk["theta"] = [[[0.9, 0.1, 0.0, 0.01]]]
    base = copy.deepcopy(one_manifest)
    base["calls"][0]["resolved"][-1]["fit_kwargs"]["theta"] = [[[1.0, 0.0, 0.0, 0.0]]]
    assert cohort_provenance([base, other])["n_runs"] == 2
