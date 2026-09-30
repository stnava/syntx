"""
Benchmark provenance: everything needed to know -- and reproduce -- how a result was made.

Motivation: the 2026-09-20 90-pair cohort result was stored without parameters, and the
script that generated it was never committed, so its settings could only be reconstructed
afterwards (and the reconstruction conflicts with the committed code). A manifest built
here travels with every benchmark result, so that cannot happen again:

- ``code``:        git commit, branch, dirty flag, the full uncommitted diff (tracked files)
                   and its sha256, untracked file names, syntx version and install path.
- ``script``:      the invoking script's path, sha256 and full text (a runner that is never
                   committed is still preserved inside its own results).
- ``environment``: host, platform, python / torch / antspyx / numpy versions, devices.
- ``calls``:       every registration call (syntx.syn / registration / tvf / syngs / greedy /
                   robust_affine, ants.registration) with the arguments it was given and --
                   for syntx models -- the *resolved* parameters it actually ran with,
                   captured at the model's ``fit()`` (hidden defaults included).

Usage::

    with capture_registration_calls() as cap:
        res = syntx.syn(fixed, moving)
    manifest = build_manifest(calls=cap.calls, run={"pair_idx": 44})
    assert_manifest_complete(manifest)
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import importlib
import os
import platform
import socket
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
REQUIRED_KEYS = ("schema_version", "timestamp_utc", "code", "script", "environment", "calls")
_MAX_TEXT = 2_000_000  # bytes of diff / script text embedded in a manifest


# ----------------------------------------------------------------------------------------
# JSON-safe conversion
# ----------------------------------------------------------------------------------------
def jsonable(v: Any, depth: int = 0) -> Any:
    """Convert a value to something ``json.dump`` accepts, summarising arrays/images."""
    if depth > 6:
        return repr(v)[:200]
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else repr(v)
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return jsonable(v.item(), depth + 1)
        if isinstance(v, np.ndarray):
            return v.tolist() if v.size <= 32 else f"<ndarray shape={v.shape} dtype={v.dtype}>"
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        if isinstance(v, torch.Tensor):
            if v.numel() <= 32:
                return v.detach().cpu().tolist()
            return f"<tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}>"
        if isinstance(v, torch.device):
            return str(v)
    except ImportError:  # pragma: no cover
        pass
    try:
        import ants
        if isinstance(v, ants.ANTsImage):
            return {"ANTsImage": {"shape": list(v.shape), "spacing": list(v.spacing),
                                  "origin": list(v.origin), "components": v.components}}
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, dict):
        return {str(k): jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x, depth + 1) for x in v]
    if callable(v):
        return f"<callable {getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', repr(v))}>"
    return repr(v)[:200]


def _is_param_like(v: Any) -> bool:
    """Scalars / short sequences of scalars: what a hyper-parameter looks like."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)) and len(v) <= 64:
        return all(x is None or isinstance(x, (bool, int, float, str)) for x in v)
    return False


# ----------------------------------------------------------------------------------------
# Code / script / environment state
# ----------------------------------------------------------------------------------------
def _git(root: str, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", root, *args], stderr=subprocess.DEVNULL,
                                       timeout=30).decode("utf-8", errors="replace")
    except Exception:
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_state(root: Optional[str] = None, include_diff: bool = True,
               pkg_dir: Optional[str] = None) -> Dict[str, Any]:
    """Git state of the syntx checkout that is actually imported (not the CWD).

    ``pkg_dir`` (default: the imported syntx package) is where untracked source files
    count as code changes; ``root`` (default: ``pkg_dir``) locates the git checkout.
    """
    import syntx
    pkg_dir = os.path.abspath(pkg_dir or os.path.dirname(os.path.abspath(syntx.__file__)))
    root = root or pkg_dir
    top = _git(root, "rev-parse", "--show-toplevel")
    state: Dict[str, Any] = {
        "syntx_version": getattr(syntx, "__version__", None),
        "syntx_path": pkg_dir,
        "git": None,
    }
    if top is None:
        return state
    top = top.strip()
    diff = _git(top, "diff", "HEAD", "--binary") or ""
    untracked = (_git(top, "ls-files", "--others", "--exclude-standard") or "").split()
    status = _git(top, "status", "--porcelain", "--untracked-files=no") or ""
    # Untracked source files inside the imported package change behaviour but are not in
    # `git diff`: they make the checkout dirty and their contents are embedded.
    pkg_rel = os.path.relpath(pkg_dir, top)
    untracked_pkg = {}
    for rel in untracked:
        if rel.startswith(pkg_rel + os.sep) and rel.endswith(".py"):
            with open(os.path.join(top, rel), "rb") as f:
                untracked_pkg[rel] = f.read()
    diff_bytes = diff.encode("utf-8") + b"".join(
        rel.encode() + b"\0" + data for rel, data in sorted(untracked_pkg.items()))
    git = {
        "root": top,
        "commit": (_git(top, "rev-parse", "HEAD") or "").strip() or None,
        "branch": (_git(top, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "describe": (_git(top, "describe", "--tags", "--always", "--dirty") or "").strip() or None,
        "dirty": bool(status.strip()) or bool(untracked_pkg),
        "diff_sha256": _sha256(diff_bytes),  # tracked diff + untracked package sources
        "diff_bytes": len(diff_bytes),
        "untracked": untracked[:500],
        "untracked_package_files": sorted(untracked_pkg),
    }
    if include_diff:
        git["diff"] = diff if len(diff.encode("utf-8")) <= _MAX_TEXT else None
        git["diff_truncated"] = len(diff.encode("utf-8")) > _MAX_TEXT
        git["untracked_package_sources"] = {
            rel: data.decode("utf-8", errors="replace") for rel, data in untracked_pkg.items()}
    state["git"] = git
    return state


def script_state() -> Dict[str, Any]:
    """The script / entry point that launched this process, with its full text."""
    argv = list(sys.argv)
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    out: Dict[str, Any] = {"argv": argv, "cwd": os.getcwd(), "path": None,
                           "sha256": None, "text": None}
    if path and os.path.isfile(path):
        path = os.path.abspath(path)
        with open(path, "rb") as f:
            data = f.read()
        out.update(path=path, sha256=_sha256(data),
                   text=data.decode("utf-8", errors="replace") if len(data) <= _MAX_TEXT else None)
    return out


# The invoking script as it was when this process started (what the process executes;
# later edits on disk do not change the running code). Benchmark runners import this module
# at start-up via syntx.benchmark.evaluate.
_PROCESS_SCRIPT = script_state()


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    for mod, name in [("torch", "torch"), ("ants", "antspyx"), ("numpy", "numpy"),
                      ("scipy", "scipy"), ("jax", "jax"), ("antstorch", "antstorch")]:
        try:
            env["packages"][name] = getattr(importlib.import_module(mod), "__version__", "unknown")
        except Exception:
            env["packages"][name] = None
    try:
        import torch
        env["devices"] = {"cuda": bool(torch.cuda.is_available()),
                          "mps": bool(getattr(torch.backends, "mps", None)
                                      and torch.backends.mps.is_available())}
    except Exception:  # pragma: no cover
        env["devices"] = None
    return env


# ----------------------------------------------------------------------------------------
# Registration-call capture
# ----------------------------------------------------------------------------------------
# (module, attribute) of every registration entry point benchmarks use.
_ENTRY_POINTS = [
    ("syntx", "syn"), ("syntx", "registration"), ("syntx", "tvf"),
    ("syntx", "tvf_registration"), ("syntx", "syngs"), ("syntx", "syngs_registration"),
    ("syntx", "greedy"), ("syntx", "greedy_registration"), ("syntx", "robust_affine"),
    ("syntx", "auto_reg"), ("ants", "registration"),
]
# (module, class) whose fit() receives the fully resolved parameters.
_MODELS = [
    ("syntx.syn", "SyNTo"), ("syntx.tvf", "TVFModel"),
    ("syntx.syngs", "GeodesicShootingModel"), ("syntx.greedy", "GreedyRegistrationModel"),
]

_state = threading.local()


def _stack() -> List[dict]:
    if not hasattr(_state, "stack"):
        _state.stack = []
    return _state.stack


class capture_registration_calls:
    """Context manager recording every registration call made inside it.

    ``cap.calls`` is a list of ``{"function", "args", "kwargs", "resolved", "status",
    "nested_in"}`` dicts. ``resolved`` holds, per model fit, the fit() keyword arguments
    and the model's hyper-parameter attributes (scalars / short lists) at fit time.
    Originals are always restored on exit, even on error. Not re-entrant across threads.
    """

    def __init__(self):
        self.calls: List[dict] = []
        self._patches: List[tuple] = []

    # -- wrappers ------------------------------------------------------------------------
    def _wrap_entry(self, qualname: str, fn):
        cap = self

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            stack = _stack()
            rec = {
                "function": qualname,
                "args": [jsonable(a) for a in args],
                "kwargs": {k: jsonable(v) for k, v in kwargs.items()},
                "resolved": [],
                "status": "running",
                "nested_in": stack[-1]["function"] if stack else None,
            }
            cap.calls.append(rec)
            stack.append(rec)
            try:
                out = fn(*args, **kwargs)
                rec["status"] = "ok"
                return out
            except BaseException as e:
                rec["status"] = f"error: {type(e).__name__}"
                raise
            finally:
                stack.pop()

        wrapper.__syntx_provenance_wrapped__ = fn
        return wrapper

    def _wrap_fit(self, qualname: str, fit):
        @functools.wraps(fit)
        def wrapper(model, *args, **kwargs):
            stack = _stack()
            if stack:
                attrs = {k: jsonable(v) for k, v in vars(model).items()
                         if not k.startswith("_") and _is_param_like(v)}
                stack[-1]["resolved"].append({
                    "model": qualname,
                    "fit_kwargs": {k: jsonable(v) for k, v in kwargs.items()},
                    "model_attributes": attrs,
                })
            return fit(model, *args, **kwargs)

        wrapper.__syntx_provenance_wrapped__ = fit
        return wrapper

    # -- patching ------------------------------------------------------------------------
    def _patch(self, owner, name, new):
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, new)

    def start(self, include_diff: bool = True) -> "capture_registration_calls":
        # Code state as of the start of the run; build_manifest(start=cap.start_state)
        # records it and flags any change on disk while the run was executing.
        self.start_state = {
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "code": code_state(include_diff=include_diff),
        }
        wrapped: Dict[int, Any] = {}  # one wrapper per function object (syn is registration)
        for mod_name, attr in _ENTRY_POINTS:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            fn = getattr(mod, attr, None)
            if fn is None or hasattr(fn, "__syntx_provenance_wrapped__"):
                continue
            if id(fn) not in wrapped:
                wrapped[id(fn)] = self._wrap_entry(f"{mod_name}.{attr}", fn)
            self._patch(mod, attr, wrapped[id(fn)])
        for mod_name, cls_name in _MODELS:
            try:
                cls = getattr(importlib.import_module(mod_name), cls_name)
            except (ImportError, AttributeError):
                continue
            fit = cls.__dict__.get("fit")
            if fit is None or hasattr(fit, "__syntx_provenance_wrapped__"):
                continue
            self._patch(cls, "fit", self._wrap_fit(f"{mod_name}.{cls_name}", fit))
        return self

    def stop(self) -> None:
        while self._patches:
            owner, name, orig = self._patches.pop()
            setattr(owner, name, orig)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# ----------------------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------------------
def _fingerprint(code: Dict[str, Any]) -> Dict[str, Any]:
    g = code.get("git") or {}
    return {"commit": g.get("commit"), "dirty": g.get("dirty"), "diff_sha256": g.get("diff_sha256")}


def build_manifest(calls: Optional[List[dict]] = None, run: Optional[Dict[str, Any]] = None,
                   include_diff: bool = True, start: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """Assemble a complete provenance manifest (JSON-serialisable).

    ``code`` is the state at ``start`` (``capture_registration_calls().start_state``) when
    given, else now; ``script`` is the invoking script as of process start. If the code or
    script on disk changed while the run executed, ``changed_during_run`` is True and the
    end-of-run fingerprints are kept under ``at_end``.
    """
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    code = start["code"] if start else code_state(include_diff=include_diff)
    script = _PROCESS_SCRIPT if _PROCESS_SCRIPT.get("path") else script_state()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": now,
        "started_utc": start["timestamp_utc"] if start else now,
        "code": code,
        "script": script,
        "environment": environment(),
        "run": jsonable(run or {}),
        "calls": calls or [],
        "changed_during_run": False,
    }
    end_code = _fingerprint(code_state(include_diff=False)) if start else _fingerprint(code)
    end_script = script_state().get("sha256") if script.get("path") else None
    if end_code != _fingerprint(code) or (end_script and end_script != script.get("sha256")):
        manifest["changed_during_run"] = True
        manifest["at_end"] = {"code": end_code, "script_sha256": end_script}
    return manifest


def with_provenance(evaluator: Optional[str] = None):
    """Decorator for benchmark evaluators returning a result dict.

    Captures every registration call made while the evaluator runs and stores the full
    manifest under ``result["provenance"]``; the evaluator's bound arguments (defaults
    applied) are recorded under ``manifest["run"]``.
    """
    import inspect

    def deco(fn):
        sig = inspect.signature(fn)
        name = evaluator or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            with capture_registration_calls() as cap:
                result = fn(*args, **kwargs)
            if isinstance(result, dict):
                result["provenance"] = build_manifest(
                    calls=cap.calls, run={"evaluator": name, **bound.arguments},
                    start=cap.start_state)
            return result

        return wrapper

    return deco


def assert_manifest_complete(manifest: Dict[str, Any], require_calls: bool = True) -> None:
    """Raise ValueError if a manifest cannot identify the code and parameters of a run."""
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"provenance manifest missing keys: {missing}")
    git = (manifest.get("code") or {}).get("git")
    if not git or not git.get("commit"):
        raise ValueError("provenance manifest has no git commit (run from a git checkout)")
    if git.get("dirty") and not (git.get("diff") or git.get("diff_sha256")):
        raise ValueError("dirty checkout but no diff recorded")
    if require_calls:
        top = [c for c in manifest["calls"] if c.get("nested_in") is None]
        if not top:
            raise ValueError("provenance manifest records no registration call")
        for c in top:
            if c["function"].startswith("syntx.") and c["function"] != "syntx.robust_affine" \
                    and not c.get("resolved"):
                raise ValueError(f"{c['function']} call has no resolved parameters")


def resolved_parameters(manifest: Dict[str, Any], function_prefix: str = "syntx.") -> Dict[str, Any]:
    """Flatten the first matching top-level call: explicit kwargs + resolved values.

    Returns ``{"function", "explicit", "fit_kwargs", "model_attributes"}``.
    """
    for c in manifest.get("calls", []):
        if c.get("nested_in") is None and c["function"].startswith(function_prefix) \
                and c["function"] != "syntx.robust_affine":
            res = c["resolved"][-1] if c.get("resolved") else {}
            return {"function": c["function"], "explicit": c["kwargs"],
                    "fit_kwargs": res.get("fit_kwargs", {}),
                    "model_attributes": res.get("model_attributes", {})}
    return {}


# fit() keywords that carry per-image data rather than parameters (initial affine / grid)
_IMAGE_DEPENDENT_FIT_KEYS = ("initial_grid", "theta")


def cohort_provenance(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise per-pair manifests for a cohort record, refusing inconsistent inputs.

    All manifests must come from the same commit, the same uncommitted diff and the same
    resolved parameters (image-dependent values excluded); otherwise ValueError.
    """
    if not manifests:
        raise ValueError("no manifests")
    for m in manifests:
        assert_manifest_complete(m)
        if m.get("changed_during_run"):
            raise ValueError("a run's code/script changed on disk while it executed; rerun it")
    keys = set()
    for m in manifests:
        g = m["code"]["git"]
        keys.add((g["commit"], g["diff_sha256"]))
    registration_code = None
    if len(keys) != 1:
        # Runs from different clean commits are one cohort only if the registration code is
        # byte-identical (commits that changed tuner / provenance code only), as the tuner's
        # cache assumes; anything else is refused.
        import hashlib as _h
        from syntx.benchmark.tune import registration_code_fingerprint
        empty = _h.sha256(b"").hexdigest()
        if any(d != empty for _, d in keys):
            raise ValueError(f"cohort mixes code states: {sorted(keys)}")
        fps = {registration_code_fingerprint(commit=c) for c, _ in keys}
        if len(fps) != 1 or None in fps:
            raise ValueError(f"cohort mixes code states: {sorted(keys)}")
        registration_code = fps.pop()

    def _params(m):
        p = resolved_parameters(m)
        # drop per-image values (shapes, spacings, origins) from model attributes
        attrs = {k: v for k, v in p.get("model_attributes", {}).items()
                 if k not in ("grid_shape", "spacing", "origin", "direction", "moving_shape",
                              "moving_spacing", "moving_origin", "moving_direction",
                              "image_shape", "velocity_shape")
                 and not k.endswith("_shape")}
        # drop per-image inputs: image tensors/arrays and the per-pair initial affine
        fit = {k: v for k, v in p.get("fit_kwargs", {}).items()
               if not k.startswith(("fixed_", "moving_", "init_"))
               and k not in _IMAGE_DEPENDENT_FIT_KEYS
               and not (isinstance(v, str) and v.startswith(("<tensor", "<ndarray", "{'ANTsImage'")))
               and not (isinstance(v, dict) and "ANTsImage" in v)}
        exp = {k: v for k, v in p.get("explicit", {}).items()
               if k not in ("fixed", "moving", "initial_transform")}
        return {"function": p.get("function"), "explicit": exp, "fit_kwargs": fit,
                "model_attributes": attrs}

    params = [_params(m) for m in manifests]
    for p in params[1:]:
        if p != params[0]:
            diff = {k for k in params[0] if params[0][k] != p[k]}
            raise ValueError(f"cohort mixes parameter sets (differ in {sorted(diff)})")
    first = manifests[0]
    g = first["code"]["git"]
    return {
        "schema_version": SCHEMA_VERSION,
        "n_runs": len(manifests),
        "commit": g["commit"],
        "commits": sorted({c for c, _ in keys}),
        "registration_code": registration_code,
        "describe": g.get("describe"),
        "dirty": g["dirty"],
        "diff_sha256": g["diff_sha256"],
        "syntx_version": first["code"].get("syntx_version"),
        "script_sha256": sorted({m["script"].get("sha256") for m in manifests
                                 if m["script"].get("sha256")}),
        "hosts": sorted({m["environment"].get("hostname") for m in manifests}),
        "packages": first["environment"].get("packages"),
        "parameters": params[0],
    }


# ----------------------------------------------------------------------------------------
# Published result records (docs/provenance/best_parameters.json)
# ----------------------------------------------------------------------------------------
def record_result(path: str, record_key: str, metrics: Dict[str, Any], manifests,
                  method_key: Optional[str] = None) -> Dict[str, Any]:
    """Add a result record whose provenance is derived from the runs' own manifests.

    ``manifests`` is a list of per-run manifests (single-method record) or a dict
    ``{arm_name: [manifests]}`` (multi-arm record; ``metrics`` should then hold
    ``{"arms": {arm_name: {...}}}``). Records are stored at ``data[method_key][record_key]``
    (e.g. method_key="syntx.syn") or top-level ``data[record_key]``. Existing records are
    never overwritten. Returns the stored record.
    """
    import json

    if isinstance(manifests, dict):
        prov = {"arms": {arm: cohort_provenance(ms) for arm, ms in manifests.items()}}
        if "arms" in metrics:
            missing = set(metrics["arms"]) - set(manifests)
            if missing:
                raise ValueError(f"arms without manifests: {sorted(missing)}")
    else:
        prov = cohort_provenance(list(manifests))
    record = {**jsonable(metrics), "provenance": prov}

    with open(path) as f:
        data = json.load(f)
    container = data.setdefault(method_key, {}) if method_key else data
    if record_key in container:
        raise ValueError(f"record {method_key + '/' if method_key else ''}{record_key} exists")
    container[record_key] = record
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    os.replace(tmp, path)
    return record


def check_record_provenance(record: Dict[str, Any]) -> None:
    """Raise ValueError unless a result record carries per-run-derived provenance."""
    prov = record.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("record has no provenance")
    blocks = prov["arms"].items() if "arms" in prov else [("record", prov)]
    if "arms" in record and "arms" not in prov:
        raise ValueError("multi-arm record needs provenance per arm")
    for arm in (record.get("arms") or {}):
        if arm not in dict(blocks):
            raise ValueError(f"arm {arm!r} has no provenance")
    for arm, p in blocks:
        commit = p.get("commit") or ""
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError(f"{arm}: provenance has no full git commit")
        if not p.get("parameters") or not p.get("n_runs") or "diff_sha256" not in p:
            raise ValueError(f"{arm}: provenance lacks parameters / n_runs / diff_sha256")
