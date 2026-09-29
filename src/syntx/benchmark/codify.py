"""
Codify a tuning result (``syntx.benchmark.tune``): commit the winning parameters as the
method's new defaults on a **branch**, never on the current branch.

Steps (all automatic):
1. create a git worktree on branch ``tune/<method>-<YYYYMMDD>`` from the commit the tune ran on;
2. rewrite the method's signature defaults (located via the AST, so only the literal default
   expression is replaced) and the parameter's "Default ..." docstring line;
3. rewrite the matching keys of the method's ``DEFAULT_BENCHMARK_CONFIG`` block;
4. add the tuning report as ``docs/provenance/tuning/<method>_<date>.md``;
5. run the method's canonical-parameter and unit tests inside the worktree;
6. commit (and push) the branch only if they pass; remove the worktree.

Parameters that are not signature defaults (resolved in a function body) cannot be codified
automatically; codify refuses rather than committing a partial change.
"""

from __future__ import annotations

import ast
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

# method -> where its defaults live (paths relative to the repo root)
TARGETS: Dict[str, Dict[str, Any]] = {
    "greedy": {
        "function_file": "src/syntx/greedy.py",
        "function": "greedy_registration",
        "config_block": "greedy_config",
        # tuned parameter -> config.py key
        "config_keys": {"learning_rate": "grad_step", "flow_sigma": "flow_sigma",
                        "total_sigma": "total_sigma", "optimizer": "optimizer",
                        "regadam_sigma": "regadam_sigma", "reg_iterations": "reg_iterations"},
        # hidden (body-resolved) defaults whose authoritative value lives only in config.py
        "config_only": {"reg_iterations"},
        "tests": ["tests/test_canonical_greedy_parameters.py", "tests/test_greedy.py",
                  "tests/test_tune.py"],
    },
}


class CodifyError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------
# Source rewriting (pure functions; unit-tested)
# ----------------------------------------------------------------------------------------
def _offsets(src: str):
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return lambda lineno, col: starts[lineno - 1] + col


def _replace_spans(src: str, spans: List[tuple]) -> str:
    for start, end, text in sorted(spans, reverse=True):
        src = src[:start] + text + src[end:]
    return src


def rewrite_function_defaults(src: str, function: str, values: Dict[str, Any]) -> str:
    """Return ``src`` with the default expressions of ``function``'s parameters replaced."""
    tree = ast.parse(src)
    off = _offsets(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == function), None)
    if fn is None:
        raise CodifyError(f"function {function!r} not found")
    a = fn.args
    positional = a.posonlyargs + a.args
    pairs = list(zip(positional[len(positional) - len(a.defaults):], a.defaults))
    pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    by_name = {arg.arg: d for arg, d in pairs}
    missing = sorted(set(values) - set(by_name))
    if missing:
        raise CodifyError(f"{function}: {missing} are not parameters with defaults")
    spans = []
    for name, value in values.items():
        d = by_name[name]
        spans.append((off(d.lineno, d.col_offset), off(d.end_lineno, d.end_col_offset), repr(value)))
    out = _replace_spans(src, spans)
    ast.parse(out)
    return out


def rewrite_docstring_defaults(src: str, function: str, old: Dict[str, Any],
                               new: Dict[str, Any]) -> str:
    """Best-effort: in ``function``'s numpydoc parameter entry, 'Default <old>' -> 'Default <new>'."""
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function), None)
    if fn is None or not (fn.body and isinstance(fn.body[0], ast.Expr)):
        return src
    doc_node = fn.body[0].value
    off = _offsets(src)
    start, end = off(doc_node.lineno, doc_node.col_offset), off(doc_node.end_lineno, doc_node.end_col_offset)
    doc = src[start:end]
    for name, value in new.items():
        m = re.search(rf"(?m)^([ \t]*){re.escape(name)} :.*\n(?:\1[ \t]+.*(?:\n|$))*", doc)
        if not m or name not in old:
            continue
        entry = m.group(0)
        entry2 = re.sub(rf"Default {re.escape(str(old[name]))}(?!\d)", f"Default {value}", entry, count=1)
        doc = doc.replace(entry, entry2, 1)
    return src[:start] + doc + src[end:]


def rewrite_config_block(src: str, block: str, values: Dict[str, Any]) -> str:
    """Replace ``values`` in the ``DEFAULT_BENCHMARK_CONFIG[block]`` dict literal."""
    tree = ast.parse(src)
    off = _offsets(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == block and isinstance(v, ast.Dict):
                    target = v
    if target is None:
        raise CodifyError(f"config block {block!r} not found")
    spans, found = [], set()
    for k, v in zip(target.keys, target.values):
        if isinstance(k, ast.Constant) and k.value in values:
            spans.append((off(v.lineno, v.col_offset), off(v.end_lineno, v.end_col_offset),
                          repr(values[k.value])))
            found.add(k.value)
    missing = sorted(set(values) - found)
    if missing:
        raise CodifyError(f"config block {block!r} lacks keys {missing}")
    out = _replace_spans(src, spans)
    ast.parse(out)
    return out


# ----------------------------------------------------------------------------------------
# Git plumbing
# ----------------------------------------------------------------------------------------
def _git(root, *args, check=True):
    return subprocess.run(["git", "-C", root, *args], check=check, capture_output=True, text=True)


def _repo_root() -> str:
    import syntx
    return _git(os.path.dirname(os.path.abspath(syntx.__file__)), "rev-parse",
                "--show-toplevel").stdout.strip()


def apply_to_tree(root: str, method: str, winner: Dict[str, Any], defaults: Dict[str, Any],
                  report_text: Optional[str] = None, date: Optional[str] = None,
                  targets: Optional[Dict[str, Any]] = None) -> List[str]:
    """Write ``winner`` (parameter -> new default) into the tree at ``root``; returns changed files."""
    t = targets or TARGETS.get(method)
    if t is None:
        raise CodifyError(f"no codify targets registered for method {method!r}")
    unmapped = sorted(set(winner) - set(t["config_keys"]))
    if unmapped:
        raise CodifyError(f"cannot codify {unmapped} for {method}: no known default location")
    changed = []
    sig_values = {k: v for k, v in winner.items() if k not in t.get("config_only", set())}
    if sig_values:
        path = os.path.join(root, t["function_file"])
        with open(path) as f:
            src = f.read()
        new = rewrite_function_defaults(src, t["function"], sig_values)
        new = rewrite_docstring_defaults(new, t["function"], defaults, sig_values)
        with open(path, "w") as f:
            f.write(new)
        changed.append(t["function_file"])
    cfg_path = os.path.join(root, "src/syntx/benchmark/config.py")
    cfg_values = {t["config_keys"][k]: v for k, v in winner.items()}
    with open(cfg_path) as f:
        cfg_src = f.read()
    cfg_new = rewrite_config_block(cfg_src, t["config_block"], cfg_values)  # read fully before writing
    with open(cfg_path, "w") as f:
        f.write(cfg_new)
    changed.append("src/syntx/benchmark/config.py")
    if report_text:
        date = date or _dt.datetime.now().strftime("%Y-%m-%d")
        rel = f"docs/provenance/tuning/{method}_{date}.md"
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        open(os.path.join(root, rel), "w").write(report_text)
        changed.append(rel)
    return changed


def codify(result: Dict[str, Any], out_dir: Optional[str] = None, push: bool = True,
           run_tests: bool = True, repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Commit ``result``'s winning parameters as new defaults on a branch (see module doc)."""
    method = result["method"]
    winner = result["winner"]["overrides"]
    if not winner:
        raise CodifyError("tuning found no confirmed improvement; nothing to codify")
    root = repo_root or _repo_root()
    base = (result.get("code") or {}).get("commit")
    if not base:
        raise CodifyError("tuning result has no code commit")
    date = _dt.datetime.now().strftime("%Y%m%d")
    branch = f"tune/{method}-{date}"
    n = 1
    while _git(root, "rev-parse", "--verify", "--quiet", branch, check=False).returncode == 0:
        n += 1
        branch = f"tune/{method}-{date}-{n}"
    wt = tempfile.mkdtemp(prefix=f"syntx_codify_{method}_")
    os.rmdir(wt)
    _git(root, "worktree", "add", "-q", "-b", branch, wt, base)
    try:
        report = None
        if out_dir and os.path.exists(os.path.join(out_dir, "report.md")):
            report = open(os.path.join(out_dir, "report.md")).read()
        changed = apply_to_tree(wt, method, winner, result["defaults"], report,
                                date=_dt.datetime.now().strftime("%Y-%m-%d"))
        tests_ok, test_log = True, ""
        if run_tests:
            tests = [p for p in TARGETS[method]["tests"] if os.path.exists(os.path.join(wt, p))]
            env = dict(os.environ, PYTHONPATH=os.path.join(wt, "src"))
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                                  cwd=wt, env=env, capture_output=True, text=True)
            tests_ok, test_log = proc.returncode == 0, proc.stdout[-4000:]
        if not tests_ok:
            return {"branch": None, "committed": False, "tests_passed": False,
                    "test_log": test_log, "changed": changed}
        _git(wt, "add", *changed)
        lines = [f"tune({method}): new defaults from automated tuning ({date})", ""]
        lines += [f"- {k}: {result['defaults'].get(k)!r} -> {v!r}" for k, v in sorted(winner.items())]
        lines += ["", f"Mean Dice gain {result['winner']['gain']:+.4f} over pairs {result['pairs']} "
                  f"(margin {result['margin']:.5f}); per-pair constraints satisfied, winner "
                  f"confirmed by a repeat run. Tuned at {base[:10]}; record: "
                  f"{result.get('record_key', 'n/a')}.", "",
                  "Generated by syntx.benchmark.codify; review and merge."]
        _git(wt, "commit", "-q", "-m", "\n".join(lines))
        commit = _git(wt, "rev-parse", "HEAD").stdout.strip()
        pushed = False
        if push:
            pushed = _git(wt, "push", "-q", "-u", "origin", branch, check=False).returncode == 0
        return {"branch": branch, "commit": commit, "committed": True, "pushed": pushed,
                "tests_passed": tests_ok, "test_log": test_log, "changed": changed}
    finally:
        _git(root, "worktree", "remove", "--force", wt, check=False)
