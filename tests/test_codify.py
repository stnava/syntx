"""
syntx.benchmark.codify: AST-exact rewriting of signature defaults / config blocks /
docstrings, and the branch-commit flow (worktree, commit on a new branch, original
branch and working tree untouched).
"""

import os
import shutil
import subprocess

import pytest

from syntx.benchmark.codify import (
    CodifyError,
    apply_to_tree,
    codify,
    record_canonical,
    rewrite_config_block,
    rewrite_dict_constant,
    rewrite_docstring_defaults,
    rewrite_function_defaults,
    rewrite_json_block,
    targets_for,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC = '''def f(a, b=0.25, *, c="adam", d=[1, 2]):
    """Doc.

    Parameters
    ----------
    b : float, optional
        Step size. Default 0.25.
    c : str, optional
        Optimizer. Default 'adam'.
    """
    return b
'''


def test_rewrite_function_defaults_is_exact():
    out = rewrite_function_defaults(SRC, "f", {"b": 0.4, "c": "regadam", "d": [3]})
    assert "def f(a, b=0.4, *, c='regadam', d=[3]):" in out
    assert out.count("0.25") == 1  # docstring untouched by this step
    with pytest.raises(CodifyError, match="not parameters"):
        rewrite_function_defaults(SRC, "f", {"zzz": 1})


def test_rewrite_docstring_defaults():
    out = rewrite_docstring_defaults(SRC, "f", {"b": 0.25}, {"b": 0.4})
    assert "Step size. Default 0.4." in out and "Default 'adam'" in out


def test_rewrite_config_block():
    cfg = 'X = {"greedy_config": {"grad_step": 0.25, "flow_sigma": 1.8}, "other": {"grad_step": 9}}\n'
    out = rewrite_config_block(cfg, "greedy_config", {"grad_step": 0.4})
    assert '"greedy_config": {"grad_step": 0.4, "flow_sigma": 1.8}' in out and '"grad_step": 9' in out
    with pytest.raises(CodifyError, match="lacks keys"):
        rewrite_config_block(cfg, "greedy_config", {"nope": 1})


def test_apply_to_tree_on_real_greedy_sources(tmp_path):
    for rel in ("src/syntx/greedy.py", "src/syntx/benchmark/config.py"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    changed = apply_to_tree(str(tmp_path), "greedy", {"learning_rate": 0.4, "flow_sigma": 2.2},
                            {"learning_rate": 0.25, "flow_sigma": 1.8}, report_text="# r\n",
                            date="2026-01-01")
    g = (tmp_path / "src/syntx/greedy.py").read_text()
    assert "learning_rate: float = 0.4," in g and "flow_sigma: float = 2.2," in g
    c = (tmp_path / "src/syntx/benchmark/config.py").read_text()
    assert '"grad_step": 0.4' in c and '"flow_sigma": 2.2' in c
    assert "docs/provenance/tuning/greedy_2026-01-01.md" in changed
    with pytest.raises(CodifyError, match="cannot codify"):
        apply_to_tree(str(tmp_path), "greedy", {"unknown_param": 1}, {})


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()


def test_codify_commits_on_a_branch_and_leaves_checkout_alone(tmp_path):
    repo = tmp_path / "repo"
    for rel in ("src/syntx/greedy.py", "src/syntx/benchmark/config.py"):
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    base = _git(repo, "rev-parse", "HEAD")
    out_dir = tmp_path / "tune"
    out_dir.mkdir()
    (out_dir / "report.md").write_text("# Tuning report: greedy\n")
    result = {"method": "greedy", "pairs": [77, 44, 0], "margin": 0.0005,
              "defaults": {"learning_rate": 0.25}, "code": {"commit": base},
              "winner": {"overrides": {"learning_rate": 0.4}, "gain": 0.003}}
    env_email = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    old = {k: os.environ.get(k) for k in env_email}
    os.environ.update(env_email)
    try:
        out = codify(result, out_dir=str(out_dir), push=False, run_tests=False, repo_root=str(repo),
                     record=False)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert out["committed"] and out["branch"].startswith("tune/greedy-")
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"       # still on main
    assert _git(repo, "rev-parse", "HEAD") == base                          # main unchanged
    assert _git(repo, "status", "--porcelain") == ""                       # checkout clean
    shown = _git(repo, "show", f"{out['branch']}:src/syntx/greedy.py")
    assert "learning_rate: float = 0.4," in shown
    assert "learning_rate: 0.25 -> 0.4" in _git(repo, "log", "-1", "--format=%B", out["branch"])
    assert _git(repo, "worktree", "list").count("\n") == 0                 # worktree removed

    with pytest.raises(CodifyError, match="no confirmed improvement"):
        codify(dict(result, winner={"overrides": {}, "gain": 0.0}), push=False, repo_root=str(repo))


def test_rewrite_json_block_keeps_formatting():
    text = '{\n    "a": {\n        "x": 0.50,\n        "l": [100, 100, 20],\n        "s": "adam"\n    },\n    "b": {"x": 9}\n}\n'
    out = rewrite_json_block(text, "a", {"x": 0.375, "s": "regadam"})
    assert '"x": 0.375,' in out and '"s": "regadam"' in out and '"l": [100, 100, 20]' in out
    assert '"b": {"x": 9}' in out and out.count("\n") == text.count("\n")
    with pytest.raises(CodifyError, match="lacks keys"):
        rewrite_json_block(text, "a", {"zz": 1})


def test_rewrite_dict_constant():
    src = "D = {2: 0.060, 3: 0.45}\nX = 1\n"
    assert rewrite_dict_constant(src, "D", 3, 0.6) == "D = {2: 0.060, 3: 0.6}\nX = 1\n"
    with pytest.raises(CodifyError):
        rewrite_dict_constant(src, "D", 4, 1.0)


def test_targets_come_from_the_method_declarations():
    from syntx.benchmark.tune import METHODS
    for m in METHODS:
        t = targets_for(m)
        assert t["function_file"] and t["config_block"]
        assert "tests/test_canonical_parameters.py" in t["tests"]


def test_apply_to_tree_places_each_default_where_it_lives(tmp_path):
    """syngs: max_step_norm is a signature default, alpha lives in SYNGS_DEFAULT_ALPHA[3];
    config.py and run_config.json follow."""
    for rel in ("src/syntx/syngs.py", "src/syntx/benchmark/config.py", "docs/provenance/run_config.json"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    changed = apply_to_tree(str(tmp_path), "syngs", {"alpha": 0.6, "max_step_norm": 0.25},
                            {"alpha": 0.45, "max_step_norm": 0.19})
    g = (tmp_path / "src/syntx/syngs.py").read_text()
    assert "SYNGS_DEFAULT_ALPHA = {2: 0.060, 3: 0.6}" in g and "max_step_norm=0.25," in g
    c = (tmp_path / "src/syntx/benchmark/config.py").read_text()
    assert '"alpha": 0.6' in c and '"max_step_norm": 0.25' in c
    import json
    rc = json.load(open(tmp_path / "docs/provenance/run_config.json"))["syngs_config"]
    assert rc["alpha"] == 0.6 and rc["max_step_norm"] == 0.25
    assert set(changed) == {"src/syntx/syngs.py", "src/syntx/benchmark/config.py",
                            "docs/provenance/run_config.json"}


def test_record_canonical_writes_record_and_pointer(tmp_path):
    import json
    import numpy as np
    import ants
    import syntx
    from syntx.provenance import build_manifest, capture_registration_calls
    arr = np.random.default_rng(0).random((16, 16, 16)).astype(np.float32)
    f, m = ants.from_numpy(arr), ants.from_numpy(np.roll(arr, 1, 0))
    with capture_registration_calls() as cap:
        syntx.greedy(fixed=f, moving=m, initial_transform=False, reg_iterations=[1, 1, 1], device="cpu")
    man = build_manifest(calls=cap.calls, run={}, include_diff=False)
    out = tmp_path / "tune"
    (out / "runs").mkdir(parents=True)
    rows = []
    for p in (77, 44, 0):
        json.dump({"provenance": man}, open(out / "runs" / f"r{p}.json", "w"))
        rows.append({"method": "greedy", "pair": p, "rep": 0, "overrides": {"learning_rate": 0.4},
                     "record_file": f"runs/r{p}.json", "metrics": {}})
    with open(out / "evaluations.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    root = tmp_path / "repo"
    (root / "docs" / "provenance").mkdir(parents=True)
    json.dump({"syntx.greedy": {}}, open(root / "docs/provenance/best_parameters.json", "w"))
    result = {"method": "greedy", "pairs": [77, 44, 0], "defaults": {"learning_rate": 0.375},
              "winner": {"overrides": {"learning_rate": 0.4}, "gain": 0.002, "per_pair": {}},
              "baseline": {}, "criteria": {}, "margin": 0.0005}
    changed = record_canonical(str(root), "greedy", result, str(out))
    best = json.load(open(root / "docs/provenance/best_parameters.json"))["syntx.greedy"]
    (key, rec), = best.items()
    assert key.startswith("canonical_") and rec["parameters"]["learning_rate"] == 0.4
    assert rec["provenance"]["n_runs"] == 3
    ptr = json.load(open(root / "docs/provenance/canonical.json"))
    assert ptr["greedy"] == f"syntx.greedy/{key}"
    assert "docs/provenance/canonical.json" in changed
