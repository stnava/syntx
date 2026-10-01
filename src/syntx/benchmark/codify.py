"""
Codify a tuning result (``syntx.benchmark.tune``): commit the winning parameters as the
method's new defaults on a **branch**, never on the current branch.

Steps (``codify``):
1. create a git worktree on a new branch ``tune/<method>-<YYYYMMDD>`` (``-2``, ``-3``, ... if
   taken) from the commit the tune ran on (``result['code']['commit']``);
2. rewrite the method's signature defaults (located via the AST, so only the default
   expression is replaced) and, best effort, the parameter's "Default <old>" docstring text;
   parameters declared in ``MethodSpec.constant_defaults`` are rewritten in that module-level
   dict constant instead;
3. rewrite the matching keys of the method's ``DEFAULT_BENCHMARK_CONFIG`` block in
   ``config.py`` and of ``docs/provenance/run_config.json`` (keys present there only);
4. add the tuning report (``<out_dir>/report.md``) as
   ``docs/provenance/tuning/<method>_<YYYY-MM-DD>.md``;
5. optionally add a ``best_parameters.json`` record and point ``docs/provenance/canonical.json``
   at it;
6. run tests/test_canonical_parameters.py, tests/test_tune.py and the method's own tests
   inside the worktree (pytest with the current interpreter, ``PYTHONPATH=<worktree>/src``);
7. commit, and push to ``origin``, only if they pass; the worktree is removed in all cases.

Locations come from the method's declaration in ``syntx.benchmark.tune.METHODS``. A winning
parameter that is neither a declared constant nor in the declaration's ``config_keys`` (even if
it is a signature default) cannot be codified: ``apply_to_tree`` raises before writing
anything.
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

def targets_for(method: str) -> Dict[str, Any]:
    """Where ``method``'s defaults live, read from its ``syntx.benchmark.tune.METHODS``
    declaration (the same declaration tests/test_canonical_parameters.py checks).

    Returns
    -------
    dict
        'function_file' (repo-relative), 'function', 'config_block', 'run_config_block',
        'config_keys' (inverted: function parameter -> config key), 'constant_defaults'
        (parameter -> (file, constant name, dict key)), 'tests' (the two shared test files
        plus the method's own).

    Raises
    ------
    CodifyError
        Unknown method or one without a ``function_file``.
    """
    from syntx.benchmark.tune import METHODS
    spec = METHODS.get(method)
    if spec is None or not spec.function_file:
        raise CodifyError(f"no canonical declaration for method {method!r}")
    return {
        "function_file": spec.function_file,
        "function": spec.function_name,
        "config_block": spec.config_block,
        "run_config_block": spec.run_config_block,
        "config_keys": {param: key for key, param in spec.config_keys.items()},  # param -> key
        "constant_defaults": dict(spec.constant_defaults),
        "tests": ("tests/test_canonical_parameters.py", "tests/test_tune.py") + tuple(spec.tests),
    }


class CodifyError(RuntimeError):
    """A tuning result cannot be codified (unknown location, missing key, nothing to commit)."""
    pass


# ----------------------------------------------------------------------------------------
# Source rewriting (pure functions; unit-tested)
# ----------------------------------------------------------------------------------------
def _offsets(src: str):
    """Return ``f(lineno, col) -> character offset`` into ``src`` (AST line numbers are 1-based)."""
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return lambda lineno, col: starts[lineno - 1] + col


def _replace_spans(src: str, spans: List[tuple]) -> str:
    """Apply ``(start, end, text)`` replacements, last span first so offsets stay valid."""
    for start, end, text in sorted(spans, reverse=True):
        src = src[:start] + text + src[end:]
    return src


def rewrite_function_defaults(src: str, function: str, values: Dict[str, Any]) -> str:
    """Return ``src`` with the default expressions of ``function``'s parameters replaced.

    The first (sync or async) function named ``function`` anywhere in the module is used.
    Each new default is written as ``repr(value)``; the result is re-parsed as a check.

    Parameters
    ----------
    src : str
        Module source.
    function : str
        Function name.
    values : dict
        Parameter name -> new default value.

    Raises
    ------
    CodifyError
        Function not found, or a name that is not a parameter with a default.
    """
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
    """Best effort: in each numpydoc entry ``<name> : ...`` of ``function``'s docstring, replace
    the first 'Default <old>' with 'Default <new>' (``str`` formatting). Entries that are
    missing, or names without an ``old`` value, are left alone; with no function or no
    docstring, ``src`` is returned unchanged. Only plain (non-async) functions are searched."""
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
    """Replace ``values`` in the dict literal stored under key ``block`` (as written in
    ``DEFAULT_BENCHMARK_CONFIG``), writing each as ``repr(value)``. If several dict literals
    have a ``block`` key, the last one found by ``ast.walk`` is used.

    Raises
    ------
    CodifyError
        Block not found, or a key of ``values`` absent from it.
    """
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


def rewrite_json_block(text: str, block: str, values: Dict[str, Any]) -> str:
    """Replace ``values`` in the object ``block`` of a JSON document, keeping the file's
    formatting (only the value tokens change).

    Text-based: the first ``"block": {`` occurrence is used (not necessarily top level), and
    each key's first occurrence inside it is replaced by ``json.dumps(value)``. A value is
    matched as a flat list, a string or a scalar token, so keys holding nested objects or
    nested lists are not supported. The result is checked with ``json.loads``.

    Raises
    ------
    CodifyError
        Block or key not found.
    """
    import json
    import re
    m = re.search(r'"%s"\s*:\s*\{' % re.escape(block), text)
    if not m:
        raise CodifyError(f"JSON block {block!r} not found")
    start, depth, i = m.end(), 1, m.end()
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    body = text[start:i - 1]
    missing = []
    for k, v in values.items():
        pat = re.compile(r'("%s"\s*:\s*)(\[[^\]]*\]|"(?:[^"\\]|\\.)*"|[^,\n}\]]+)' % re.escape(k))
        if not pat.search(body):
            missing.append(k)
            continue
        body = pat.sub(lambda mm: mm.group(1) + json.dumps(v), body, count=1)
    if missing:
        raise CodifyError(f"JSON block {block!r} lacks keys {missing}")
    out = text[:start] + body + text[i - 1:]
    json.loads(out)
    return out


def rewrite_dict_constant(src: str, name: str, key: Any, value: Any) -> str:
    """Replace the value for ``key`` in the module-level-style assignment
    ``NAME = {..., key: <old>, ...}`` with ``repr(value)`` (AST-located).

    Raises
    ------
    CodifyError
        Constant or key not found.
    """
    tree = ast.parse(src)
    off = _offsets(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    out = _replace_spans(src, [(off(v.lineno, v.col_offset),
                                                off(v.end_lineno, v.end_col_offset), repr(value))])
                    ast.parse(out)
                    return out
            raise CodifyError(f"{name} has no key {key!r}")
    raise CodifyError(f"dict constant {name} not found")


# ----------------------------------------------------------------------------------------
# Git plumbing
# ----------------------------------------------------------------------------------------
def _git(root, *args, check=True):
    """``git -C root <args>`` (captured text output; raises CalledProcessError if ``check``)."""
    return subprocess.run(["git", "-C", root, *args], check=check, capture_output=True, text=True)


def _repo_root() -> str:
    """Top-level directory of the git checkout that contains the imported ``syntx`` package."""
    import syntx
    return _git(os.path.dirname(os.path.abspath(syntx.__file__)), "rev-parse",
                "--show-toplevel").stdout.strip()


def apply_to_tree(root: str, method: str, winner: Dict[str, Any], defaults: Dict[str, Any],
                  report_text: Optional[str] = None, date: Optional[str] = None,
                  targets: Optional[Dict[str, Any]] = None) -> List[str]:
    """Write ``winner`` (parameter -> new default) into the source tree at ``root``.

    Each parameter is placed by its declaration: a ``constant_defaults`` entry -> that dict
    constant (only supported in ``function_file``); a signature default that is also in
    ``config_keys`` -> the signature (and its "Default ..." docstring text); a ``config_keys``
    entry that is not a signature parameter -> config only. Every parameter in
    ``config_keys`` is additionally written to ``src/syntx/benchmark/config.py`` (block
    ``config_block``) and, for keys already present there, to
    ``docs/provenance/run_config.json`` (block ``run_config_block``).

    Parameters
    ----------
    root : str
        Repository root to modify (codify passes a temporary worktree).
    method : str
        Method name in ``syntx.benchmark.tune.METHODS``.
    winner : dict
        Parameter (function keyword) -> new default.
    defaults : dict
        Previous defaults, used to find the "Default <old>" docstring text.
    report_text : str, optional
        Written to ``docs/provenance/tuning/<method>_<date>.md``.
    date : str, optional
        Date for the report name; default today, 'YYYY-MM-DD'.
    targets : dict, optional
        Override of ``targets_for(method)`` (used by tests).

    Returns
    -------
    list of str
        Repo-relative paths of the files written.

    Raises
    ------
    CodifyError
        A parameter with no known location (raised before anything is written), or a
        missing function / constant / config key (files already written earlier in the call
        stay written).
    """
    import inspect as _inspect
    t = targets or targets_for(method)
    with open(os.path.join(root, t["function_file"])) as f:
        fn_src = f.read()
    fn_node = next((n for n in ast.walk(ast.parse(fn_src))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == t["function"]), None)
    if fn_node is None:
        raise CodifyError(f"function {t['function']!r} not found in {t['function_file']}")
    a = fn_node.args
    sig_params = {x.arg for x in (a.posonlyargs + a.args)[len(a.posonlyargs + a.args) - len(a.defaults):]}
    sig_params |= {x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}
    const = t.get("constant_defaults", {})
    placements = {}
    for k in winner:
        if k in const:
            placements[k] = "constant"
        elif k in sig_params and k in t["config_keys"]:
            placements[k] = "signature"
        elif k in t["config_keys"] and k not in sig_params:
            placements[k] = "config_only"
    unmapped = sorted(set(winner) - set(placements))
    if unmapped:
        raise CodifyError(f"cannot codify {unmapped} for {method}: no known default location")

    changed = []
    sig_values = {k: v for k, v in winner.items() if placements[k] == "signature"}
    new_src = fn_src
    if sig_values:
        new_src = rewrite_function_defaults(new_src, t["function"], sig_values)
        new_src = rewrite_docstring_defaults(new_src, t["function"], defaults, sig_values)
    for k, v in winner.items():
        if placements[k] == "constant":
            f_rel, name, key = const[k]
            if f_rel != t["function_file"]:
                raise CodifyError(f"constant for {k!r} lives in another file ({f_rel}); unsupported")
            new_src = rewrite_dict_constant(new_src, name, key, v)
    if new_src != fn_src:
        with open(os.path.join(root, t["function_file"]), "w") as f:
            f.write(new_src)
        changed.append(t["function_file"])

    cfg_values = {t["config_keys"][k]: v for k, v in winner.items() if k in t["config_keys"]}
    if cfg_values:
        cfg_path = os.path.join(root, "src/syntx/benchmark/config.py")
        with open(cfg_path) as f:
            cfg_src = f.read()
        cfg_new = rewrite_config_block(cfg_src, t["config_block"], cfg_values)  # read fully first
        with open(cfg_path, "w") as f:
            f.write(cfg_new)
        changed.append("src/syntx/benchmark/config.py")
        rc_path = os.path.join(root, "docs/provenance/run_config.json")
        if t.get("run_config_block") and os.path.exists(rc_path):
            with open(rc_path) as f:
                rc = f.read()
            import json as _json
            block = _json.loads(rc).get(t["run_config_block"], {})
            rc_values = {k: v for k, v in cfg_values.items() if k in block}
            if rc_values:
                with open(rc_path, "w") as f:
                    f.write(rewrite_json_block(rc, t["run_config_block"], rc_values))
                changed.append("docs/provenance/run_config.json")
    if report_text:
        date = date or _dt.datetime.now().strftime("%Y-%m-%d")
        rel = f"docs/provenance/tuning/{method}_{date}.md"
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as f:
            f.write(report_text)
        changed.append(rel)
    return changed


def record_canonical(root: str, method: str, result: Dict[str, Any], out_dir: str) -> List[str]:
    """In the tree at ``root``: add a ``docs/provenance/best_parameters.json`` record for the
    codified winner and point ``docs/provenance/canonical.json`` at it.

    The record key is ``canonical_<YYYY_MM_DD>`` (``_2``, ``_3``, ... if taken) under
    ``syntx.<method>``; its metrics hold the full parameter set (defaults updated by the
    winner), the previous defaults, the mean Dice gain, per-pair and baseline metrics,
    criteria and margin; its provenance comes from the winner's rep-0 run manifests in
    ``out_dir`` (``winner_manifests_from_dir``; KeyError if any pair is missing).

    Returns
    -------
    list of str
        The two repo-relative paths written.
    """
    import json as _json
    from syntx.benchmark.tune import CANONICAL_POINTER, winner_manifests_from_dir
    from syntx.provenance import record_result
    best_path = os.path.join(root, "docs/provenance/best_parameters.json")
    with open(best_path) as f:
        existing = _json.load(f).get(f"syntx.{method}", {})
    key = f"canonical_{_dt.datetime.now().strftime('%Y_%m_%d')}"
    n = 1
    while key in existing:
        n += 1
        key = f"canonical_{_dt.datetime.now().strftime('%Y_%m_%d')}_{n}"
    params = dict(result["defaults"])
    params.update(result["winner"]["overrides"])
    manifests = winner_manifests_from_dir(out_dir, method, result["winner"]["overrides"], result["pairs"])
    metrics = {"selection": "codified tuning winner (syntx.benchmark.codify)", "pairs": result["pairs"],
               "parameters": params, "previous_defaults": result["defaults"],
               "mean_dice_gain_vs_previous_defaults": result["winner"]["gain"],
               "per_pair": result["winner"]["per_pair"], "baseline_per_pair": result["baseline"],
               "criteria": result.get("criteria"), "margin": result.get("margin")}
    record_result(best_path, key, metrics, manifests, method_key=f"syntx.{method}")
    ptr_path = os.path.join(root, CANONICAL_POINTER)
    ptr = _json.load(open(ptr_path)) if os.path.exists(ptr_path) else {}
    ptr[method] = f"syntx.{method}/{key}"
    with open(ptr_path, "w") as f:
        _json.dump(ptr, f, indent=4)
        f.write("\n")
    return ["docs/provenance/best_parameters.json", CANONICAL_POINTER]


def codify(result: Dict[str, Any], out_dir: Optional[str] = None, push: bool = True,
           run_tests: bool = True, repo_root: Optional[str] = None,
           record: bool = True) -> Dict[str, Any]:
    """Commit ``result``'s winning parameters as new defaults on a new branch (see module doc).

    Parameters
    ----------
    result : dict
        A ``syntx.benchmark.tune.tune`` / ``Tuner.run`` result. Uses 'method',
        'winner' ('overrides', 'gain', 'per_pair'), 'defaults', 'fixed_parameters', 'code'
        ('commit'), 'pairs', 'margin', 'baseline', 'criteria' and optionally 'record_key'.
    out_dir : str, optional
        The tune's output directory; supplies report.md and, with ``record``, the run
        manifests. Without it, neither the report nor the record is added.
    push : bool, default True
        ``git push -u origin <branch>`` after committing (failure only sets 'pushed' False).
    run_tests : bool, default True
        Run the tests before committing; False commits untested.
    repo_root : str, optional
        Repository to branch from; default the checkout containing ``syntx``.
    record : bool, default True
        Add the best_parameters / canonical.json record (``record_canonical``).

    Returns
    -------
    dict
        On success: 'branch', 'commit', 'committed' (True), 'pushed', 'tests_passed',
        'test_log' (last 4000 characters of pytest output), 'changed'. If the tests fail:
        'branch' None, 'committed' False, 'tests_passed' False, 'test_log', 'changed'; the
        branch created for the worktree is left in the repository pointing at the base commit.

    Raises
    ------
    CodifyError
        Empty winner, a winning parameter that was fixed during the tune, no code commit in
        ``result``, or a placement error from ``apply_to_tree``.
    """
    method = result["method"]
    winner = result["winner"]["overrides"]
    if not winner:
        raise CodifyError("tuning found no confirmed improvement; nothing to codify")
    fixed = sorted(set(winner) & set(result.get("fixed_parameters") or {}))
    if fixed:
        raise CodifyError(f"{fixed} were fixed during this tune and cannot be codified from it")
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
        if record and out_dir:
            changed += record_canonical(wt, method, result, out_dir)
        tests_ok, test_log = True, ""
        if run_tests:
            tests = [p for p in targets_for(method)["tests"] if os.path.exists(os.path.join(wt, p))]
            env = dict(os.environ, PYTHONPATH=os.path.join(wt, "src"))
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                                  cwd=wt, env=env, capture_output=True, text=True)
            tests_ok, test_log = proc.returncode == 0, proc.stdout[-4000:]
        if not tests_ok:
            return {"branch": None, "committed": False, "tests_passed": False,
                    "test_log": test_log, "changed": changed}
        _git(wt, "add", *sorted(set(changed)))
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
