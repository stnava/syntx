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
    rewrite_config_block,
    rewrite_docstring_defaults,
    rewrite_function_defaults,
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
        out = codify(result, out_dir=str(out_dir), push=False, run_tests=False, repo_root=str(repo))
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
