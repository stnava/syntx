"""
Automated parameter tuning for syntx registration methods: search -> refine -> validate ->
report -> record (-> codify, see ``syntx.benchmark.codify``).

This systematises what used to be done by hand (docs/provenance/syn_param_sweep_mbhard_*.md):

1. **Baseline & noise** -- the method's current defaults are run twice on every pair; the
   run-to-run difference sets the acceptance margin ``max(min_gain, noise_k * noise)``.
2. **Screen** -- one-at-a-time changes over the method's search space.
3. **Refine** -- constrained hill-climb from the best feasible point: pairwise / all-top-k
   combinations of improving moves plus bracketing of every changed numeric parameter;
   a new best is accepted only if it beats the current best by more than the margin
   *and* a repeat run confirms it.
4. **Validate** -- built in: the objective is the mean Dice over all pairs (default
   Mindboggle 77, 44, 0) and every pair must satisfy the constraints relative to the
   baseline, including a maximum per-pair Dice drop (a single search pair overfits).
5. **Report / record** -- ranked Markdown + JSON of every configuration; optionally a
   ``best_parameters.json`` record via ``syntx.provenance.record_result`` whose provenance
   is derived from the winning runs' manifests.

Every evaluation is cached on disk (``evaluations.jsonl``) keyed by method, pair, parameter
overrides, repeat index and code state, so an interrupted tune resumes where it stopped.
The checkout must be clean, runs must not change code mid-run, and each pair's affine is
held constant (its sha256 is checked on every evaluation).

    from syntx.benchmark.tune import tune
    result = tune("greedy")                       # pairs (77, 44, 0)
    python -m syntx.benchmark.tune --method greedy --isolate --record --codify
    python -m syntx.benchmark.tune --watch results/tune_greedy_<date>     # monitor live
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import itertools
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_PAIRS: Tuple[int, ...] = (77, 44, 0)
# Held fixed during a tune unless the caller unfixes them (Tuner(fixed_parameters=...),
# CLI --unfix NAME): the benchmark schedule is not a regularisation choice.
DEFAULT_FIXED_PARAMETERS: Dict[str, Any] = {"reg_iterations": [100, 100, 20]}
FIXED_PARAMETERS = DEFAULT_FIXED_PARAMETERS  # backwards-compatible name
AFFINE_CACHE = "results/canonical_affines/pair_{pair:03d}_pt7_affine.mat"


# ----------------------------------------------------------------------------------------
# Search space
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class Param:
    """One tunable parameter.

    Numeric parameters without explicit ``values`` are screened on a multiplicative ladder
    around the current value (``factors``), clipped to [lo, hi]. ``requires`` makes the
    parameter active only when other parameters take given values, e.g.
    ``requires={"optimizer": ("regadam",)}``.
    """
    name: str
    kind: str = "float"                      # "float" | "int" | "categorical" | "list"
    values: Optional[Sequence[Any]] = None
    factors: Sequence[float] = (0.5, 0.75, 1.5, 2.0)
    lo: Optional[float] = None
    hi: Optional[float] = None
    requires: Optional[Dict[str, Sequence[Any]]] = None

    def active(self, config: Dict[str, Any]) -> bool:
        if not self.requires:
            return True
        return all(config.get(k) in tuple(v) for k, v in self.requires.items())

    def _clip(self, v):
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        if self.kind == "int":
            v = int(round(v))
        return round(v, 6) if isinstance(v, float) else v

    def candidates(self, current: Any) -> List[Any]:
        if self.values is not None:
            vals = list(self.values)
        elif self.kind in ("float", "int") and isinstance(current, (int, float)) and current:
            vals = [self._clip(current * f) for f in self.factors]
        else:
            vals = []
        out = []
        for v in vals:
            if v != current and v not in out:
                out.append(v)
        return out

    def bracket(self, best: Any, tried: Sequence[Any]) -> List[Any]:
        """Values between ``best`` and its nearest tried neighbours, and one step beyond."""
        if self.kind not in ("float", "int") or not isinstance(best, (int, float)):
            return []
        nums = sorted({float(t) for t in tried if isinstance(t, (int, float))} | {float(best)})
        i = nums.index(float(best))
        out = []
        if i > 0:
            out.append(self._clip((nums[i - 1] + best) / 2.0))
        else:
            out.append(self._clip(best * 0.75))
        if i < len(nums) - 1:
            out.append(self._clip((nums[i + 1] + best) / 2.0))
        else:
            out.append(self._clip(best * 1.33))
        return [v for v in out if v != best and v not in tried]


@dataclasses.dataclass
class MethodSpec:
    """How to run and tune one registration method through the standard evaluator."""
    name: str
    model: str                                # evaluate_mindboggle_pair(model=...)
    function: str                             # provenance name of the registration call
    defaults: Callable[[], Dict[str, Any]]    # effective defaults of the tunable parameters
    space: List[Param]
    has_inverse: bool = True
    # keyword arguments the evaluator may pass without it being a parameter override
    evaluator_plumbing: Tuple[str, ...] = ("fixed", "moving", "initial_transform", "backend",
                                           "device", "verbose")


def _signature_defaults(func_path: str, names: Sequence[str], hidden: Dict[str, Any] = None):
    def get():
        import importlib
        import inspect
        mod_name, attr = func_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(mod_name), attr)
        sig = inspect.signature(fn)
        out = {n: sig.parameters[n].default for n in names if n in sig.parameters}
        out.update(hidden or {})
        return out
    return get


METHODS: Dict[str, MethodSpec] = {
    "greedy": MethodSpec(
        name="greedy", model="greedy", function="syntx.greedy",
        defaults=_signature_defaults(
            "syntx.greedy.greedy_registration",
            ["learning_rate", "flow_sigma", "total_sigma", "optimizer", "regadam_sigma",
             "lncc_radius", "reg_iterations"],
            hidden={"reg_iterations": [100, 100, 20]}),
        space=[
            Param("learning_rate", lo=0.05, hi=1.0),
            Param("flow_sigma", lo=0.5, hi=6.0),
            Param("total_sigma", values=(0.0, 0.14, 0.42, 0.56, 0.8)),
            Param("optimizer", kind="categorical", values=("adam", "regadam")),
            Param("regadam_sigma", lo=0.2, hi=3.0, requires={"optimizer": ("regadam", "reg_adam")}),
            Param("lncc_radius", kind="int", values=(1, 3)),
            # fixed by default (DEFAULT_FIXED_PARAMETERS); searched only if unfixed
            Param("reg_iterations", kind="list", values=([100, 100, 50], [100, 50, 10], [200, 100, 20])),
        ],
        has_inverse=False,   # greedy does not generate an inverse: inverse metrics are NaN
    ),
    "syn": MethodSpec(
        name="syn", model="sobolev", function="syntx.syn",
        defaults=_signature_defaults(
            "syntx.syn.registration",
            ["grad_step", "flow_sigma", "total_sigma", "fast_smooth", "in_loop_inv_steps",
             "optimizer"],
            hidden={"regularizer": "sobolev", "sobolev_alpha": 1.5, "fast_smooth": False,
                    "reg_iterations": [100, 100, 20]}),
        space=[
            Param("grad_step", lo=0.05, hi=0.75),
            Param("flow_sigma", lo=1.0, hi=8.0),
            Param("sobolev_alpha", lo=0.5, hi=4.0,
                  requires={"regularizer": ("sobolev", "dsti", "dsti1")}),
            Param("regularizer", kind="categorical", values=("sobolev", "dsti1", "gaussian")),
            Param("fast_smooth", kind="categorical", values=(False, True)),
            Param("optimizer", kind="categorical", values=("cfl", "reg_adam")),
        ],
    ),
}


# ----------------------------------------------------------------------------------------
# Constraints / scoring
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class Criteria:
    """Feasibility is judged per pair *relative to the baseline* (current defaults): even
    the defaults fold slightly / exceed 1 mm inverse error on some pairs."""
    folding_abs: float = 0.0005        # folding % always allowed up to this ...
    folding_rel: float = 1.0           # ... or up to baseline folding * folding_rel
    inv_interior_abs: float = 1.0      # interior max inverse error (mm) allowed up to this ...
    inv_interior_rel: float = 1.1      # ... or up to baseline * inv_interior_rel
    positive_jac_if_baseline: bool = True
    # The inverse cap is relative to the baseline's inverse error, which is not a meaningful
    # reference where the baseline itself folds: apply it only on fold-free baseline pairs.
    inverse_cap_only_if_baseline_fold_free: bool = True
    max_pair_drop: float = 0.001       # no pair's Dice may fall more than this below baseline
    min_gain: float = 0.0005
    noise_k: float = 3.0


METRIC_KEYS = {
    "dice_sym": "syntx_dice_sym", "dice_fixed": "syntx_dice_fixed", "dice_moving": "syntx_dice_moving",
    "folding_pct": "syntx_fold", "jac_min": "syntx_min_jac", "jac_max": "syntx_max_jac",
    "inv_mean_mm": "syntx_inv_mean", "inv_p95_mm": "syntx_inv_p95", "inv_max_mm": "syntx_inv_max",
    "inv_interior_mean_mm": "syntx_inv_interior_mean", "inv_interior_max_mm": "syntx_inv_interior_max",
    "time_s": "syntx_time",
}


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def pair_violations(m: Dict[str, float], base: Dict[str, float], c: Criteria,
                    has_inverse: bool) -> List[str]:
    out = []
    if m["dice_sym"] < base["dice_sym"] - c.max_pair_drop:
        out.append(f"dice drop {base['dice_sym'] - m['dice_sym']:.4f} > {c.max_pair_drop}")
    fold_cap = max(c.folding_abs, base["folding_pct"] * c.folding_rel)
    if m["folding_pct"] > fold_cap + 1e-12:
        out.append(f"folding {m['folding_pct']:.4f}% > {fold_cap:.4f}%")
    if c.positive_jac_if_baseline and base["jac_min"] > 0 and not m["jac_min"] > 0:
        out.append("jacobian min reached 0 (baseline > 0)")
    inverse_applies = not (c.inverse_cap_only_if_baseline_fold_free and base["folding_pct"] > 0)
    if has_inverse and inverse_applies and not _nan(base["inv_interior_max_mm"]):
        cap = max(c.inv_interior_abs, base["inv_interior_max_mm"] * c.inv_interior_rel)
        if _nan(m["inv_interior_max_mm"]) or m["inv_interior_max_mm"] > cap:
            out.append(f"interior inverse max {m['inv_interior_max_mm']:.2f} > {cap:.2f} mm")
    return out


# ----------------------------------------------------------------------------------------
# Evaluation (cached, provenance-checked)
# ----------------------------------------------------------------------------------------
class NotCanonicalError(RuntimeError):
    """The evaluator passes tuning parameters the tuner did not ask for (a drifted path)."""


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _affine_sha(pair: int) -> Optional[str]:
    p = AFFINE_CACHE.format(pair=pair)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def mindboggle_evaluator(spec: MethodSpec, generate_report: bool = False, **fixed_kwargs):
    """Default evaluator: ``evaluate_mindboggle_pair(pair, spec.model, **overrides)``.

    Returns ``(metrics, record)``; only parameters that differ from the method's defaults
    are passed, so the baseline is exactly the standard benchmark path.
    """
    from syntx.benchmark.evaluate import evaluate_mindboggle_pair

    def run(pair: int, overrides: Dict[str, Any], report_dir: Optional[str] = None):
        rec = evaluate_mindboggle_pair(pair, spec.model, generate_report=generate_report,
                                       report_out_dir=report_dir, verbose=False,
                                       **fixed_kwargs, **overrides)
        metrics = {k: _num(rec.get(v)) for k, v in METRIC_KEYS.items()}
        if not spec.has_inverse:
            for k in ("inv_mean_mm", "inv_p95_mm", "inv_max_mm", "inv_interior_mean_mm",
                      "inv_interior_max_mm"):
                metrics[k] = float("nan")
        return metrics, rec

    return run


def check_canonical_call(spec: MethodSpec, record: Dict[str, Any], overrides: Dict[str, Any]):
    """Raise NotCanonicalError if the evaluator passed tuning keywords beyond ``overrides``."""
    from syntx.provenance import resolved_parameters
    man = record.get("provenance")
    if not man:
        raise NotCanonicalError("evaluation record has no provenance manifest")
    p = resolved_parameters(man, function_prefix=spec.function)
    if not p:
        raise NotCanonicalError(f"no {spec.function} call recorded")
    extra = sorted(set(p["explicit"]) - set(spec.evaluator_plumbing) - set(overrides))
    if extra:
        raise NotCanonicalError(
            f"the {spec.model!r} evaluator path passes {extra} to {spec.function} itself; "
            f"it must call the method with its own defaults (see syntx.benchmark.evaluate)")


# Files that do not affect registration results: tuner / codify / provenance
# instrumentation. Changing them keeps cached evaluations valid.
TUNER_ONLY_FILES = ("benchmark/tune.py", "benchmark/codify.py", "provenance.py")
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()
_FP_CACHE: Dict[str, Optional[str]] = {}


def _normalise_source(rel: str, data: bytes) -> bytes:
    if rel == "__init__.py":  # version bumps do not change registration
        data = b"\n".join(l for l in data.split(b"\n") if not l.lstrip().startswith(b"__version__"))
    return data


def registration_code_fingerprint(commit: Optional[str] = None, pkg_dir: Optional[str] = None) -> Optional[str]:
    """sha256 over the syntx package sources that can affect registration results.

    Excludes TUNER_ONLY_FILES and the ``__version__`` line. ``commit=None`` reads the
    working tree (tracked files + untracked package sources); otherwise the files at
    ``commit``. Returns None if the package is not in a git checkout / commit unknown.
    """
    import subprocess
    if pkg_dir is None:
        import syntx
        pkg_dir = os.path.dirname(os.path.abspath(syntx.__file__))
    pkg_dir = os.path.realpath(pkg_dir)  # git reports resolved paths (/var -> /private/var on macOS)
    ck = f"{pkg_dir}@{commit}"
    if commit is not None and ck in _FP_CACHE:
        return _FP_CACHE[ck]

    def git(*a):
        return subprocess.run(["git", "-C", pkg_dir, *a], capture_output=True, check=True).stdout

    try:
        root = os.path.realpath(git("rev-parse", "--show-toplevel").decode().strip())
        rel_pkg = os.path.relpath(pkg_dir, root)
        if commit is None:
            names = git("ls-files", "--cached", "--others", "--exclude-standard", "--", ".").decode().split("\n")
            files = {}
            for n in filter(None, names):
                full = os.path.join(pkg_dir, n)
                if os.path.isfile(full) and "__pycache__" not in n:
                    with open(full, "rb") as f:
                        files[n] = f.read()
        else:
            def git_root(*a):
                return subprocess.run(["git", "-C", root, *a], capture_output=True, check=True).stdout
            names = git_root("ls-tree", "-r", "--name-only", commit, "--", rel_pkg).decode().split("\n")
            files = {os.path.relpath(n, rel_pkg): git_root("show", f"{commit}:{n}") for n in filter(None, names)}
            if not files:
                raise RuntimeError(f"no package files at {commit}")
    except Exception:
        if commit is not None:
            _FP_CACHE[ck] = None
        return None
    h = hashlib.sha256()
    for rel in sorted(files):
        if rel in TUNER_ONLY_FILES or "__pycache__" in rel:
            continue
        h.update(rel.encode() + b"\0" + _normalise_source(rel, files[rel]) + b"\0")
    fp = h.hexdigest()
    if commit is not None:
        _FP_CACHE[ck] = fp
    return fp


class EvalCache:
    def __init__(self, path: str):
        self.path = path
        self.rows: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[row["key"]] = row

    def get(self, key):
        return self.rows.get(key)

    def put(self, row):
        self.rows[row["key"]] = row
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


# ----------------------------------------------------------------------------------------
# Tuner
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    overrides: Dict[str, Any]
    stage: str
    per_pair: Dict[int, List[Dict[str, float]]] = dataclasses.field(default_factory=dict)

    @property
    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.overrides.items())) or "defaults"

    def mean_metric(self, pair: int, k: str) -> float:
        vals = [r[k] for r in self.per_pair[pair] if not _nan(r[k])]
        return sum(vals) / len(vals) if vals else float("nan")

    def summary(self, pairs) -> Dict[str, float]:
        return {p: {k: self.mean_metric(p, k) for k in METRIC_KEYS} for p in pairs}


class Tuner:
    def __init__(self, method: str, pairs: Sequence[int] = DEFAULT_PAIRS,
                 space: Optional[List[Param]] = None, criteria: Criteria = None,
                 evaluator: Optional[Callable] = None, out_dir: Optional[str] = None,
                 max_evals: int = 300, max_hours: Optional[float] = None,
                 refine_rounds: int = 6, top_k: int = 3, allow_dirty: bool = False,
                 check_canonical: bool = True, code_fingerprint: Optional[Dict] = None,
                 log: Optional[Callable[[str], None]] = None,
                 fixed_parameters: Optional[Dict[str, Any]] = None):
        self.spec = METHODS[method] if isinstance(method, str) else method
        self.pairs = list(pairs)
        self.fixed = dict(DEFAULT_FIXED_PARAMETERS if fixed_parameters is None else fixed_parameters)
        space = space if space is not None else self.spec.space
        self.space = [p for p in space if p.name not in self.fixed]  # fixed ones are not searched
        self.criteria = criteria or Criteria()
        self.evaluator = evaluator or mindboggle_evaluator(self.spec)
        stamp = _dt.datetime.now().strftime("%Y%m%d")
        self.out_dir = out_dir or f"results/tune_{self.spec.name}_{stamp}"
        os.makedirs(os.path.join(self.out_dir, "runs"), exist_ok=True)
        self.cache = EvalCache(os.path.join(self.out_dir, "evaluations.jsonl"))
        self.max_evals, self.max_hours = max_evals, max_hours
        self.refine_rounds, self.top_k = refine_rounds, top_k
        self.check_canonical = check_canonical
        self.log = log or (lambda msg: print(msg, flush=True))
        self.n_new = 0
        self.t0 = time.time()
        self.defaults = self.spec.defaults()
        self.configs: Dict[str, Config] = {}
        if code_fingerprint is None:
            from syntx.provenance import code_state
            g = (code_state(include_diff=False).get("git") or {})
            if g.get("dirty") and not allow_dirty:
                raise RuntimeError("tuning requires a clean committed checkout "
                                   "(commit first, or pass allow_dirty=True)")
            code_fingerprint = {"commit": g.get("commit"), "diff_sha256": g.get("diff_sha256"),
                                "registration_code": registration_code_fingerprint()}
        self.code = code_fingerprint
        # cache identity: the registration code, not the commit (tuner-only edits keep the cache)
        self.reg_code = code_fingerprint.get("registration_code") or _key(code_fingerprint)
        self.affine = {}

    # -- evaluation ------------------------------------------------------------------------
    def _budget_left(self) -> bool:
        if self.n_new >= self.max_evals:
            return False
        if self.max_hours is not None and (time.time() - self.t0) / 3600.0 > self.max_hours:
            return False
        return True

    def evaluate(self, overrides: Dict[str, Any], stage: str, rep: int = 0) -> Optional[Config]:
        """Evaluate ``overrides`` (relative to the defaults) on every pair; cached."""
        bad = sorted(set(overrides) & set(self.fixed))
        if bad:
            raise ValueError(f"{bad} are fixed for this tune ({self.fixed}); pass "
                             f"fixed_parameters to unfix them")
        overrides = {k: v for k, v in overrides.items() if self.defaults.get(k, object()) != v}
        label_key = _key(overrides)
        cfg = self.configs.setdefault(label_key, Config(dict(overrides), stage))
        for pair in self.pairs:
            runs = cfg.per_pair.setdefault(pair, [])
            if len(runs) > rep:
                continue
            key = self._cache_key(pair, overrides, rep)
            row = self._cached(pair, overrides, rep)
            if row is None:
                if not self._budget_left():
                    return None
                self.log(f"[{time.strftime('%H:%M:%S')}] {stage} pair {pair} rep {rep}: "
                         f"{cfg.label}")
                metrics, record = self.evaluator(pair, dict(overrides))
                if record is not None:
                    self._check_record(record, overrides, pair)
                    with open(os.path.join(self.out_dir, "runs", f"{key[:16]}.json"), "w") as f:
                        json.dump(record, f, default=str)
                row = {"key": key, "method": self.spec.name, "pair": pair, "overrides": overrides,
                       "rep": rep, "stage": stage, "metrics": metrics, "code": self.code,
                       "affine_sha256": _affine_sha(pair),
                       "record_file": f"runs/{key[:16]}.json" if record is not None else None}
                self.cache.put(row)
                self.n_new += 1
                self.log("   " + ", ".join(f"{k}={metrics[k]:.4f}" for k in
                                           ("dice_sym", "folding_pct", "jac_min",
                                            "inv_interior_max_mm", "time_s")))
            self._check_affine(pair, row.get("affine_sha256"))
            runs.append(row["metrics"])
            self.stage = stage
            self.write_live()
        return cfg

    # -- real-time monitoring ----------------------------------------------------------------
    def write_live(self):
        """Rewrite live.md (progress + ranking so far) and live.csv (one row per evaluation).

        Monitor with ``python -m syntx.benchmark.tune --watch <out_dir>``.
        """
        elapsed = (time.time() - self.t0) / 60.0
        rate = elapsed / self.n_new if self.n_new else float("nan")
        lines = [f"# LIVE tuning: {self.spec.name} -- updated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                 f"- stage: **{getattr(self, 'stage', '?')}** | new evaluations: {self.n_new}/{self.max_evals} "
                 f"| elapsed {elapsed:.1f} min | {rate:.2f} min/evaluation | pairs {self.pairs}",
                 f"- code: `{(self.code or {}).get('commit')}` | out: `{self.out_dir}`"]
        base_ok = getattr(self, "baseline", None) is not None and self._complete(self.baseline)
        if base_ok:
            if hasattr(self, "margin"):
                lines.append(f"- noise {self.noise:.5f} | acceptance margin {self.margin:.5f}")
            ranking = self.ranking()
            best = next((r for r in ranking if r["feasible"] and r["overrides"]), None)
            lines.append(f"- best feasible so far: **{best['label']}** ({best['gain']:+.4f}, "
                         f"{best['n_reps']} rep)" if best else "- best feasible so far: defaults")
            lines += ["", "| # | configuration | stage | reps | mean gain | feasible | " +
                      " | ".join(f"pair {p}: Dice / fold% / inv int max / inv max" for p in self.pairs) + " |",
                      "|---|---|---|---|---|---|" + "---|" * len(self.pairs)]
            for i, r in enumerate(ranking, 1):
                cells = []
                for p in self.pairs:
                    m = r["per_pair"][p]
                    cells.append(f"{m['dice_sym']:.4f} / {m['folding_pct']:.4f} / "
                                 f"{_mm(m['inv_interior_max_mm'])} / {_mm(m['inv_max_mm'])}")
                lines.append(f"| {i} | {r['label']} | {r['stage']} | {r['n_reps']} | {r['gain']:+.4f} | "
                             f"{'yes' if r['feasible'] else 'no'} | " + " | ".join(cells) + " |")
        tmp = os.path.join(self.out_dir, "live.md.tmp")
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, os.path.join(self.out_dir, "live.md"))
        # flat per-evaluation table
        import csv
        tmp = os.path.join(self.out_dir, "live.csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "pair", "rep", "configuration"] + list(METRIC_KEYS))
            for r in self.cache.rows.values():
                label = ", ".join(f"{k}={v}" for k, v in sorted(r["overrides"].items())) or "defaults"
                w.writerow([r.get("stage"), r["pair"], r["rep"], label] +
                           [r["metrics"].get(k) for k in METRIC_KEYS])
        os.replace(tmp, os.path.join(self.out_dir, "live.csv"))

    def _cache_key(self, pair, overrides, rep):
        return _key({"method": self.spec.name, "pair": pair, "overrides": overrides, "rep": rep,
                     "registration_code": self.reg_code})

    def _cached(self, pair, overrides, rep):
        """Cached row for (pair, overrides, rep) evaluated with the same registration code:
        by key, or a row from another clean commit whose registration code is identical."""
        row = self.cache.get(self._cache_key(pair, overrides, rep))
        if row is not None:
            return row
        want = json.dumps(overrides, sort_keys=True, default=str)
        for r in self.cache.rows.values():
            if (r.get("method") != self.spec.name or r.get("pair") != pair or r.get("rep") != rep
                    or json.dumps(r.get("overrides"), sort_keys=True, default=str) != want):
                continue
            code = r.get("code") or {}
            fp = code.get("registration_code")
            if fp is None and code.get("diff_sha256") == _EMPTY_SHA and code.get("commit"):
                fp = registration_code_fingerprint(commit=code["commit"])
            if fp is not None and fp == self.reg_code:
                return r
        return None

    def _check_record(self, record, overrides, pair):
        man = record.get("provenance") or {}
        if man.get("changed_during_run"):
            raise RuntimeError(f"code changed on disk during the pair-{pair} run; aborting")
        if self.check_canonical:
            check_canonical_call(self.spec, record, overrides)

    def _check_affine(self, pair, sha):
        if sha is None:
            return
        prev = self.affine.setdefault(pair, sha)
        if prev != sha:
            raise RuntimeError(f"pair {pair}: affine changed ({prev[:12]} -> {sha[:12]}); "
                               f"the affine must be held constant")

    # -- scoring -----------------------------------------------------------------------------
    def score(self, cfg: Config) -> Tuple[float, bool, Dict[int, List[str]]]:
        base = self.baseline.summary(self.pairs)
        s = cfg.summary(self.pairs)
        gain = sum(s[p]["dice_sym"] - base[p]["dice_sym"] for p in self.pairs) / len(self.pairs)
        viol = {p: pair_violations(s[p], base[p], self.criteria, self.spec.has_inverse)
                for p in self.pairs}
        return gain, not any(viol.values()), viol

    def _complete(self, cfg):
        return cfg is not None and all(cfg.per_pair.get(p) for p in self.pairs)

    def _value(self, overrides, name):
        return overrides.get(name, self.defaults.get(name))

    def _full(self, overrides):
        full = dict(self.defaults)
        full.update(overrides)
        return full

    # -- search ------------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        c = self.criteria
        self.log(f"tuning {self.spec.name} on pairs {self.pairs}; defaults {self.defaults}")
        self.baseline = self.evaluate({}, "baseline", rep=0)
        self.evaluate({}, "baseline", rep=1)
        if not self._complete(self.baseline) or any(len(self.baseline.per_pair[p]) < 2
                                                     for p in self.pairs):
            raise RuntimeError("budget exhausted before the baseline was measured")
        noise = sum(abs(self.baseline.per_pair[p][0]["dice_sym"] - self.baseline.per_pair[p][1]["dice_sym"])
                    for p in self.pairs) / len(self.pairs)
        self.margin = max(c.min_gain, c.noise_k * noise)
        self.noise = noise
        self.log(f"noise (mean |rep0-rep1| Dice) = {noise:.5f}; acceptance margin = {self.margin:.5f}")

        best, best_gain = {}, 0.0
        tried: Dict[str, List[Any]] = {}

        def screen(center, stage, neighbours_only=False):
            out = []
            full = self._full(center)
            for p in self.space:
                if not p.active(full):
                    continue
                cur = self._value(center, p.name)
                if neighbours_only and p.kind in ("float", "int") and p.values is None:
                    cands = [p._clip(cur * f) for f in (0.8, 1.25)]
                    cands = [v for v in cands if v != cur]
                elif neighbours_only:
                    cands = []
                else:
                    cands = p.candidates(cur)
                for v in cands:
                    ov = dict(center)
                    ov[p.name] = v
                    if not all(q.active(self._full(ov)) or q.name not in ov for q in self.space):
                        continue
                    tried.setdefault(p.name, []).append(v)
                    cfg = self.evaluate(ov, stage)
                    if self._complete(cfg):
                        out.append(cfg)
            return out

        screen({}, "screen")
        for rnd in range(1, self.refine_rounds + 1):
            if not self._budget_left():
                self.log("budget exhausted")
                break
            ranked = self.ranking()
            improving = [r for r in ranked if r["feasible"] and r["gain"] > best_gain + self.margin / 2
                         and r["overrides"] != best]
            # combinations of the top-k improving single moves (relative to the current best)
            moves = []
            for r in improving:
                delta = {k: v for k, v in r["overrides"].items() if best.get(k) != v}
                if delta and delta not in moves:
                    moves.append(delta)
                if len(moves) >= self.top_k:
                    break
            proposals = []
            for n in range(2, len(moves) + 1):
                for combo in itertools.combinations(moves, n):
                    ov = dict(best)
                    for d in combo:
                        ov.update(d)
                    proposals.append(ov)
            if not moves:
                # No feasible improving move: the incumbent sits on the Dice / topology edge,
                # which one-at-a-time moves cannot leave. Pair each Dice-gaining but infeasible
                # move (typically less regularisation) with a topology-clean move (typically
                # more regularisation) so the two can compensate.
                proposals += self.compensating_pairs(ranked, best, best_gain)
            # bracket every numeric parameter changed in the best feasible configuration
            top = next((r for r in ranked if r["feasible"]), None)
            if top is not None:
                for p in self.space:
                    if p.name in top["overrides"] and p.kind in ("float", "int"):
                        for v in p.bracket(top["overrides"][p.name], tried.get(p.name, []) +
                                           [self.defaults.get(p.name)]):
                            ov = dict(top["overrides"])
                            ov[p.name] = v
                            tried.setdefault(p.name, []).append(v)
                            proposals.append(ov)
            for ov in proposals:
                self.evaluate(ov, f"refine{rnd}")
            # accept the best feasible configuration only if it clears the margin and a repeat confirms it
            accepted = False
            for r in self.ranking():
                if not r["feasible"] or r["gain"] <= best_gain + self.margin:
                    continue
                if r["overrides"] == best:
                    continue
                cfg = self.evaluate(r["overrides"], f"confirm{rnd}", rep=1)
                if cfg is None:
                    break
                g, feas, _ = self.score(cfg)
                if feas and g > best_gain + self.margin:
                    best, best_gain, accepted = dict(r["overrides"]), g, True
                    self.log(f"round {rnd}: new best (+{g:.4f}): {cfg.label}")
                    break
                self.log(f"round {rnd}: {cfg.label} not confirmed (gain {g:.4f}, feasible {feas})")
            if not accepted:
                self.log(f"round {rnd}: no confirmed improvement; stopping")
                break
            screen(best, f"local{rnd}", neighbours_only=True)
        return self.result(best, best_gain)

    def compensating_pairs(self, ranked, best, best_gain) -> List[Dict[str, Any]]:
        """Combinations (Dice gainer) x (topology-clean move), relative to ``best``.

        gainers: infeasible configurations whose mean gain beats the best by the margin.
        clean:   configurations with no folding / Jacobian violation on any pair (they may
                 lose Dice or exceed the inverse cap), ranked by gain (least Dice loss).
        Top ``top_k`` of each; pairs that change the same parameter are skipped.
        """
        def delta(r):
            return {k: v for k, v in r["overrides"].items() if best.get(k) != v}

        def topology_clean(r):
            return all(not (v.startswith("folding") or v.startswith("jacobian"))
                       for vs in r["violations"].values() for v in vs)

        gainers = [r for r in ranked if not r["feasible"] and r["gain"] > best_gain + self.margin
                   and delta(r)][: self.top_k]
        clean = sorted((r for r in ranked if topology_clean(r) and delta(r)),
                       key=lambda r: r["gain"], reverse=True)[: self.top_k]
        out = []
        for g in gainers:
            for cl in clean:
                dg, dc = delta(g), delta(cl)
                if set(dg) & set(dc):
                    continue
                ov = dict(best)
                ov.update(dg)
                ov.update(dc)
                full = self._full(ov)
                if not all(q.active(full) or q.name not in ov for q in self.space):
                    continue
                if ov not in out:
                    out.append(ov)
        if out:
            self.log(f"compensating pairs: {len(out)} combinations of {len(gainers)} Dice gainers "
                     f"x {len(clean)} topology-clean moves")
        return out

    # -- output ------------------------------------------------------------------------------
    def ranking(self) -> List[Dict[str, Any]]:
        rows = []
        for cfg in self.configs.values():
            if not self._complete(cfg):
                continue
            gain, feas, viol = self.score(cfg)
            s = cfg.summary(self.pairs)
            rows.append({"overrides": cfg.overrides, "label": cfg.label, "stage": cfg.stage,
                         "gain": gain, "feasible": feas, "violations": viol,
                         "n_reps": min(len(cfg.per_pair[p]) for p in self.pairs),
                         "per_pair": s,
                         "mean_dice": sum(s[p]["dice_sym"] for p in self.pairs) / len(self.pairs)})
        rows.sort(key=lambda r: (r["feasible"], r["gain"]), reverse=True)
        return rows

    def result(self, best, best_gain) -> Dict[str, Any]:
        ranking = self.ranking()
        res = {
            "method": self.spec.name, "pairs": self.pairs, "defaults": self.defaults,
            "out_dir": self.out_dir, "fixed_parameters": self.fixed,
            "criteria": dataclasses.asdict(self.criteria), "noise": self.noise,
            "margin": self.margin, "code": self.code, "affine_sha256": self.affine,
            "baseline": self.baseline.summary(self.pairs),
            "winner": {"overrides": best, "gain": best_gain,
                       "parameters": self._full(best),
                       "per_pair": self.configs[_key(best)].summary(self.pairs)},
            "improved": bool(best),
            "n_configs": len(ranking), "n_new_evaluations": self.n_new,
            "ranking": ranking,
        }
        with open(os.path.join(self.out_dir, "result.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        with open(os.path.join(self.out_dir, "report.md"), "w") as f:
            f.write(render_report(res))
        self.stage = "done"
        self.write_live()
        self.log(f"wrote {self.out_dir}/report.md; winner: {best or 'defaults'} (+{best_gain:.4f})")
        return res

    def winner_manifests(self, result) -> List[Dict[str, Any]]:
        best = result["winner"]["overrides"]
        out = []
        for pair in self.pairs:
            row = self._cached(pair, best, 0)
            with open(os.path.join(self.out_dir, row["record_file"])) as f:
                out.append(json.load(f)["provenance"])
        return out


def _mm(v) -> str:
    return "NaN" if _nan(_num(v)) else f"{_num(v):.2f}"


def accumulated_table(out_dir: str, include_fixed: bool = False) -> str:
    """Markdown table of every configuration evaluated so far, read from the tune's
    ``evaluations.jsonl`` (works while a tune is running, whatever code it runs).

    One row per configuration: runs, mean Dice over pairs, gain vs the defaults, and per
    pair Dice / folding % / Jacobian min / interior max inverse / global max inverse (mm) /
    time (s), averaged over repeats. Sorted by mean Dice.
    """
    rows = []
    with open(os.path.join(out_dir, "evaluations.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    n_fixed = 0
    if not include_fixed:  # runs that touched a default-fixed parameter (e.g. an aborted earlier tune)
        kept = [r for r in rows if not set(r["overrides"]) & set(DEFAULT_FIXED_PARAMETERS)]
        n_fixed, rows = len(rows) - len(kept), kept
    if not rows:
        return "no evaluations yet\n"
    pairs = sorted({r["pair"] for r in rows}, key=lambda p: [77, 44, 0].index(p) if p in (77, 44, 0) else p)
    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = json.dumps(r["overrides"], sort_keys=True)
        g = groups.setdefault(k, {"overrides": r["overrides"], "stage": r.get("stage"), "by_pair": {}})
        g["by_pair"].setdefault(r["pair"], []).append(r["metrics"])

    def avg(ms, key):
        vals = [_num(m.get(key)) for m in ms if not _nan(_num(m.get(key)))]
        return sum(vals) / len(vals) if vals else float("nan")

    base = groups.get("{}")
    method = rows[0].get("method")
    has_inverse = METHODS[method].has_inverse if method in METHODS else True
    table = []
    for g in groups.values():
        complete = all(p in g["by_pair"] for p in pairs)
        per = {p: {k: avg(g["by_pair"][p], k) for k in METRIC_KEYS} for p in pairs if p in g["by_pair"]}
        mean = sum(per[p]["dice_sym"] for p in pairs) / len(pairs) if complete else float("nan")
        gain = float("nan")
        if complete and base and all(p in base["by_pair"] for p in pairs):
            gain = mean - sum(avg(base["by_pair"][p], "dice_sym") for p in pairs) / len(pairs)
        label = ", ".join(f"{k}={v}" for k, v in sorted(g["overrides"].items())) or "defaults"
        runs = min(len(v) for v in g["by_pair"].values())
        feas = "..."
        if complete and base and all(p in base["by_pair"] for p in pairs):
            bper = {p: {k: avg(base["by_pair"][p], k) for k in METRIC_KEYS} for p in pairs}
            viol = [v for p in pairs for v in pair_violations(per[p], bper[p], Criteria(), has_inverse)]
            feas = "yes" if not viol else "no"
        table.append((mean, label, g["stage"], runs, gain, per, complete, feas))
    table.sort(key=lambda t: (t[6], t[7] == "yes", t[0] if not _nan(t[0]) else -1), reverse=True)
    head = ("| # | configuration | stage | runs | mean Dice | gain | feasible | " +
            " | ".join(f"pair {p}: Dice / fold% / Jmin / inv int max / inv max / s" for p in pairs) + " |")
    note = f"; {n_fixed} runs touching fixed parameters hidden (include_fixed=True shows them)" if n_fixed else ""
    out = [f"# Accumulated tuning results: `{out_dir}` ({len(rows)} evaluations, {len(table)} configurations{note})",
           "(feasible = per-pair constraints vs the defaults, default Criteria; rows sorted feasible first, then mean Dice)",
           "", head, "|---|---|---|---|---|---|---|" + "---|" * len(pairs)]
    for i, (mean, label, stage, runs, gain, per, complete, feas) in enumerate(table, 1):
        cells = []
        for p in pairs:
            if p not in per:
                cells.append("...")
                continue
            m = per[p]
            cells.append(f"{m['dice_sym']:.4f} / {m['folding_pct']:.4f} / {m['jac_min']:.3f} / "
                         f"{_mm(m['inv_interior_max_mm'])} / {_mm(m['inv_max_mm'])} / {m['time_s']:.0f}")
        mean_s = "..." if not complete else f"{mean:.4f}"
        gain_s = "..." if _nan(gain) else f"{gain:+.4f}"
        out.append(f"| {i} | {label} | {stage} | {runs} | {mean_s} | {gain_s} | {feas} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_report(res: Dict[str, Any]) -> str:
    pairs = res["pairs"]
    lines = [f"# Tuning report: {res['method']}", "",
             f"- pairs: {pairs}; code: `{(res['code'] or {}).get('commit')}`",
             f"- noise (mean |rep0-rep1| Dice): {res['noise']:.5f}; acceptance margin: {res['margin']:.5f}",
             f"- criteria: {res['criteria']}",
             f"- fixed (not tuned): {res.get('fixed_parameters')}",
             f"- defaults: {res['defaults']}",
             f"- **winner**: {res['winner']['overrides'] or 'defaults (no confirmed improvement)'} "
             f"(mean Dice gain {res['winner']['gain']:+.4f})", "",
             "| rank | configuration | stage | reps | mean gain | feasible | " +
             " | ".join(f"pair {p} Dice / fold % / inv int max / inv max (mm)" for p in pairs) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(pairs)]
    for i, r in enumerate(res["ranking"], 1):
        cells = []
        for p in pairs:
            m = r["per_pair"][p]
            cells.append(f"{m['dice_sym']:.4f} / {m['folding_pct']:.4f} / "
                         f"{_mm(m['inv_interior_max_mm'])} / {_mm(m['inv_max_mm'])}")
        lines.append(f"| {i} | {r['label']} | {r['stage']} | {r['n_reps']} | {r['gain']:+.4f} | "
                     f"{'yes' if r['feasible'] else 'no'} | " + " | ".join(cells) + " |")
    lines += ["", "Violations of infeasible configurations:", ""]
    for r in res["ranking"]:
        if not r["feasible"]:
            v = {p: x for p, x in r["violations"].items() if x}
            lines.append(f"- {r['label']}: {v}")
    return "\n".join(lines) + "\n"


def tune(method: str, pairs: Sequence[int] = DEFAULT_PAIRS, record: bool = False,
         best_parameters_path: str = "docs/provenance/best_parameters.json", **kwargs) -> Dict[str, Any]:
    """Run the full automated tune for ``method``; see module docstring.

    With ``record=True`` and a confirmed winner, adds a ``best_parameters.json`` record
    (``syntx.<method>/tuned_<date>``) whose provenance comes from the winner's runs.
    """
    tuner = Tuner(method, pairs=pairs, **kwargs)
    res = tuner.run()
    if record and res["improved"]:
        from syntx.provenance import record_result
        key = f"tuned_{tuner.spec.name}_{_dt.datetime.now().strftime('%Y_%m_%d')}"
        metrics = {"pairs": list(pairs), "mean_dice_gain_vs_defaults": res["winner"]["gain"],
                   "parameters": res["winner"]["parameters"], "defaults": res["defaults"],
                   "per_pair": res["winner"]["per_pair"], "baseline_per_pair": res["baseline"],
                   "criteria": res["criteria"], "noise": res["noise"], "margin": res["margin"],
                   "tuning_report": os.path.join(tuner.out_dir, "report.md")}
        record_result(best_parameters_path, key, metrics, tuner.winner_manifests(res),
                      method_key=f"syntx.{tuner.spec.name}")
        res["record_key"] = key
    return res


def watch(out_dir: str, interval: float = 15.0, once: bool = False):
    """Print ``out_dir/live.md`` repeatedly (Ctrl-C to stop)."""
    path = os.path.join(out_dir, "live.md")
    try:
        while True:
            print("\033[2J\033[H", end="")
            print(open(path).read() if os.path.exists(path) else f"waiting for {path} ...", flush=True)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        return


def run_isolated(argv: Sequence[str]) -> int:
    """Re-run this CLI from a temporary detached worktree of the current HEAD.

    The working directory stays the repository root, so relative data / cache paths
    (examples/pairs.csv, results/...) are unchanged; only the imported code is pinned.
    """
    import subprocess
    import sys
    import tempfile
    import syntx
    pkg = os.path.dirname(os.path.abspath(syntx.__file__))
    root = subprocess.check_output(["git", "-C", pkg, "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    wt = tempfile.mkdtemp(prefix="syntx_tune_")
    os.rmdir(wt)
    subprocess.run(["git", "-C", root, "worktree", "add", "-q", "--detach", wt, head], check=True)
    try:
        env = dict(os.environ, PYTHONPATH=os.path.join(wt, "src"))
        print(f"isolated tune: code pinned at {head[:10]} in {wt}", flush=True)
        return subprocess.run([sys.executable, "-u", "-m", "syntx.benchmark.tune", *argv],
                              cwd=root, env=env).returncode
    finally:
        subprocess.run(["git", "-C", root, "worktree", "remove", "--force", wt], check=False)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Automated syntx parameter tuning")
    ap.add_argument("--method", choices=sorted(METHODS))
    ap.add_argument("--watch", metavar="OUT_DIR", default=None,
                    help="print OUT_DIR/live.md every --interval seconds (monitor a running tune)")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--table", metavar="OUT_DIR", default=None,
                    help="print the accumulated results table of OUT_DIR (running or finished tune)")
    ap.add_argument("--include-fixed", action="store_true",
                    help="with --table: also show runs that changed a default-fixed parameter")
    ap.add_argument("--pairs", type=int, nargs="+", default=list(DEFAULT_PAIRS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-evals", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--unfix", nargs="*", default=[], metavar="NAME",
                    help=f"tune parameters that are fixed by default {sorted(DEFAULT_FIXED_PARAMETERS)}")
    ap.add_argument("--codify", action="store_true",
                    help="commit the winning defaults on a branch (syntx.benchmark.codify)")
    ap.add_argument("--isolate", action="store_true",
                    help="run from a temporary detached worktree of HEAD, so the checkout can "
                         "be edited while the tune runs (edits would otherwise abort it)")
    a = ap.parse_args(argv)
    if a.table:
        print(accumulated_table(a.table, include_fixed=a.include_fixed), end="")
        return None
    if a.watch:
        return watch(a.watch, a.interval)
    if not a.method:
        ap.error("--method is required unless --watch is given")
    if a.isolate:
        return run_isolated([x for x in (argv if argv is not None else __import__("sys").argv[1:])
                             if x != "--isolate"])
    fixed = {k: v for k, v in DEFAULT_FIXED_PARAMETERS.items() if k not in set(a.unfix)}
    res = tune(a.method, pairs=a.pairs, record=a.record, out_dir=a.out,
               max_evals=a.max_evals, max_hours=a.max_hours, fixed_parameters=fixed)
    if a.codify and res["improved"]:
        from syntx.benchmark.codify import codify
        out = codify(res, out_dir=res["out_dir"])
        print(f"codify: {out}", flush=True)
    return res


if __name__ == "__main__":
    main()
