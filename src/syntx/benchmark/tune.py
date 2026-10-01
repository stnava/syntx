"""
Automated parameter tuning for syntx registration methods: search -> refine -> validate ->
report -> record (-> codify, see ``syntx.benchmark.codify``).

This systematises what used to be done by hand (docs/provenance/syn_param_sweep_mbhard_*.md).
A tune works on overrides of the method's current defaults (``METHODS[method]``); the
objective is the mean symmetric Dice over the pairs (default Mindboggle 77, 44, 0) and
feasibility is judged per pair against the baseline (``Criteria``, ``pair_violations``).

1. **Baseline & noise** -- the defaults are run twice (rep 0, rep 1) on every pair. noise =
   mean over pairs of |Dice(rep 0) - Dice(rep 1)|; acceptance margin
   ``max(min_gain, noise_k * noise)``.
2. **Screen** -- one-at-a-time changes of every active search parameter around the start
   point (``start``, default the defaults): its listed values, or a multiplicative ladder
   for numeric parameters (``Param``).
3. **Refine** (up to ``refine_rounds`` rounds) -- proposals relative to the current best
   ``best`` (initially the defaults): all combinations of two or more of the top ``top_k``
   feasible moves whose gain exceeds the best's by more than margin / 2; if there is no such
   move, "compensating pairs" (a Dice-gaining infeasible move combined with a
   topology-clean move); plus a bracketing of every numeric parameter changed in the top
   feasible configuration. Then the feasible configurations whose gain exceeds the best's by
   more than the margin are re-run (rep 1) in ranking order; the first whose two-rep mean is
   still feasible and clears the margin becomes the new best. A round with no accepted
   configuration ends the search; otherwise a local screen (x0.8 / x1.25 of
   numeric ladder parameters) around the new best follows.
4. **Report / record** -- ``result.json`` and ``report.md`` (every configuration ranked);
   optionally a ``best_parameters.json`` record via ``syntx.provenance.record_result`` whose
   provenance is derived from the winning runs' manifests.

Every evaluation is cached on disk (``<out_dir>/evaluations.jsonl``) keyed by method, pair,
overrides, repeat index and a fingerprint of the registration source code
(``registration_code_fingerprint``), so an interrupted tune resumes where it stopped and
tuner-only edits keep the cache. The dataset is not part of the key. Guards: the checkout
must be clean (unless ``allow_dirty``); a record whose provenance reports
``changed_during_run`` aborts the tune; the evaluator must not pass tuning keywords beyond
the overrides (``check_canonical_call``); and each pair's affine file must keep the sha256
first seen in this tune (not checked when the file does not exist).

Python::

    from syntx.benchmark.tune import tune
    result = tune("greedy")                       # pairs (77, 44, 0)

Command line (``main``)::

    python -m syntx.benchmark.tune --method greedy --isolate --record --codify
    python -m syntx.benchmark.tune --method syngs --start max_step_norm=0.3 --jac-min-rel 0.5
    python -m syntx.benchmark.tune --method tvf --dataset 2d          # fast 2-D pairs, CPU
    python -m syntx.benchmark.tune --watch results/tune_greedy_<date>     # monitor live
    python -m syntx.benchmark.tune --table results/tune_greedy_<date>     # results so far
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
    around the current value (``factors``), clipped to [lo, hi]; a current value of 0 or None
    gives no ladder candidates. ``requires`` makes the parameter active only when other
    parameters take given values, e.g. ``requires={"optimizer": ("regadam",)}``.

    Attributes
    ----------
    name : str
        Keyword of the registration function.
    kind : {'float', 'int', 'categorical', 'list'}, default 'float'
        'int' values are rounded after scaling; 'categorical' / 'list' use ``values`` only.
    values : sequence, optional
        Explicit candidate values (used instead of the ladder).
    factors : sequence of float, default (0.5, 0.75, 1.5, 2.0)
        Ladder multipliers.
    lo, hi : float, optional
        Clipping bounds for ladder and bracket values.
    requires : dict, optional
        ``{other parameter: allowed values}``; all must hold in the full configuration.
    """
    name: str
    kind: str = "float"                      # "float" | "int" | "categorical" | "list"
    values: Optional[Sequence[Any]] = None
    factors: Sequence[float] = (0.5, 0.75, 1.5, 2.0)
    lo: Optional[float] = None
    hi: Optional[float] = None
    requires: Optional[Dict[str, Sequence[Any]]] = None

    def active(self, config: Dict[str, Any]) -> bool:
        """True if every ``requires`` condition holds in ``config`` (defaults + overrides)."""
        if not self.requires:
            return True
        return all(config.get(k) in tuple(v) for k, v in self.requires.items())

    def _clip(self, v):
        """Clip to [lo, hi], round to int for 'int', round floats to 6 decimals."""
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        if self.kind == "int":
            v = int(round(v))
        return round(v, 6) if isinstance(v, float) else v

    def candidates(self, current: Any) -> List[Any]:
        """Screening values: ``values``, else ``current * factors`` (clipped) for a non-zero
        numeric ``current``; duplicates and ``current`` itself removed."""
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
        """Refinement values for a numeric parameter: the midpoints between ``best`` and its
        nearest lower / higher value in ``tried``, or ``best * 0.75`` / ``best * 1.33`` where
        ``best`` is the lowest / highest; clipped, excluding ``best`` and values already
        tried. Empty for non-numeric parameters."""
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
    """How to run and tune one registration method through the standard evaluator.

    Attributes
    ----------
    name : str
        Tuning name (key of ``METHODS``).
    model : str
        ``evaluate_mindboggle_pair`` model ('syn' tunes model 'sobolev').
    function : str
        Provenance name of the registration call, e.g. 'syntx.greedy'.
    defaults : callable
        Returns the defaults the overrides are relative to (signature defaults plus declared
        hidden ones; see ``_signature_defaults``).
    space : list of Param
        Search space.
    has_inverse : bool, default True
        False: inverse-error metrics are NaN and the inverse constraint is skipped.
    evaluator_plumbing : tuple of str
        Keywords the evaluator may pass to the function without them counting as overrides.
    function_file, function_name, config_block, run_config_block, config_keys, resolved,
    probe_kwargs, constant_defaults, equivalent, tests
        Canonical-default declarations (filled from ``_CANONICAL``) used by
        tests/test_canonical_parameters.py and ``syntx.benchmark.codify``: the source file /
        function holding the defaults, the ``DEFAULT_BENCHMARK_CONFIG`` and run_config.json
        blocks, config key -> function parameter, how to probe the effective value at fit()
        (``(source, key, transform)``), extra probe keywords, defaults living in a dict
        constant (``parameter -> (file, NAME, key)``), equivalent values, and the method's
        tests.
    """
    name: str
    model: str                                # evaluate_mindboggle_pair(model=...)
    function: str                             # provenance name of the registration call
    defaults: Callable[[], Dict[str, Any]]    # effective defaults of the tunable parameters
    space: List[Param]
    has_inverse: bool = True
    # keyword arguments the evaluator may pass without it being a parameter override
    evaluator_plumbing: Tuple[str, ...] = ("fixed", "moving", "initial_transform", "backend",
                                           "device", "verbose")
    # -- canonical defaults: one declaration drives tests/test_canonical_parameters.py and
    #    syntx.benchmark.codify ---------------------------------------------------------------
    # where the registration function's defaults live (for codify's signature rewrite)
    function_file: Optional[str] = None
    function_name: Optional[str] = None
    # config.py DEFAULT_BENCHMARK_CONFIG block and docs/provenance/run_config.json block;
    # config_keys maps their keys -> the registration function's parameter names
    config_block: Optional[str] = None
    run_config_block: Optional[str] = None
    config_keys: Dict[str, str] = dataclasses.field(default_factory=dict)
    # parameter -> (source, key, transform) for probing the *effective* default at fit():
    # source "fit" (fit() keyword) or "attr" (model attribute); transform maps the captured
    # value back to the parameter's convention (e.g. sigma -> variance)
    resolved: Dict[str, Tuple] = dataclasses.field(default_factory=dict)
    probe_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # defaults that live in a module-level dict constant rather than the signature:
    # parameter -> (file, CONSTANT_NAME, dict key)
    constant_defaults: Dict[str, Tuple[str, str, Any]] = dataclasses.field(default_factory=dict)
    # parameter values that are equivalent (e.g. kernel_type 'sobolev' == 'bessel')
    equivalent: Dict[str, Dict[Any, Any]] = dataclasses.field(default_factory=dict)
    tests: Tuple[str, ...] = ()


def _signature_defaults(func_path: str, names: Sequence[str], hidden: Dict[str, Any] = None):
    """Return a callable giving ``{name: signature default}`` of ``func_path`` for ``names``
    present in its signature, updated with ``hidden`` (callable values are called each time)."""
    def get():
        import importlib
        import inspect
        mod_name, attr = func_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(mod_name), attr)
        sig = inspect.signature(fn)
        out = {n: sig.parameters[n].default for n in names if n in sig.parameters}
        # hidden values may be callables, read at call time (defaults living outside the
        # signature, e.g. a per-dimension constant, must not go stale after a codify)
        out.update({k: (v() if callable(v) else v) for k, v in (hidden or {}).items()})
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
    "syngs": MethodSpec(
        name="syngs", model="syngs", function="syntx.syngs",
        defaults=_signature_defaults(
            "syntx.syngs.syngs_registration",
            ["total_sigma", "alpha", "max_step_norm", "optimizer", "optimizer_lr",
             "n_steps", "bootstrap_mode"],
            hidden={"regularizer": "sobolev", "reg_iterations": [100, 100, 20],
                    # the effective 3-D default (SYNGS_DEFAULT_ALPHA[3]): benchmarks are 3-D
                    "alpha": lambda: __import__("importlib").import_module("syntx.syngs").default_alpha(3)}),
        space=[
            Param("max_step_norm", lo=0.05, hi=0.6),
            Param("alpha", lo=0.05, hi=3.0, requires={"regularizer": ("sobolev", "dsti", "dsti1")}),
            # flow_sigma is not searched: syngs accepts it only with regularizer='gaussian' (the
            # spectral regularisers searched here use alpha); fast_smooth does not exist.
            Param("optimizer_lr", lo=0.2, hi=3.0),
            Param("n_steps", kind="int", values=(6, 12)),
            Param("regularizer", kind="categorical", values=("sobolev", "dsti1")),
            Param("bootstrap_mode", kind="categorical", values=("antithetic", "none")),
            Param("total_sigma", values=(0.0, 0.035, 0.1)),
        ],
    ),
    "tvf": MethodSpec(
        name="tvf", model="tvf", function="syntx.tvf",
        defaults=_signature_defaults(
            "syntx.tvf.tvf_registration",
            ["regularizer", "n_time_steps", "multipoint_loss", "cfl_max", "syn_sampling"],
            hidden={"reg_iterations": [100, 100, 20], "optimizer": "cfl", "grad_step": 1.0,
                    "cfl_momentum": 0.9, "energy_weight": 1e-3,
                    "total_alpha": 0.0,   # None == 0 (off)
                    "constant_speed": True,
                    "alpha": lambda: __import__("importlib").import_module("syntx.tvf").default_tvf_alpha(3)}),
        space=[
            Param("alpha", lo=0.05, hi=50.0, requires={"regularizer": ("sobolev", "dsti", "dsti1")}),
            Param("energy_weight", lo=1e-5, hi=1e-1),
            Param("grad_step", lo=0.05, hi=2.0),
            Param("total_alpha", values=(0.005, 0.02, 0.05)),
            Param("regularizer", kind="categorical", values=("sobolev", "dsti1")),
            Param("n_time_steps", kind="int", values=(2, 5)),
            Param("multipoint_loss", kind="list", values=([0.5], [0.0, 1.0])),
            Param("constant_speed", kind="categorical", values=(True, False)),
            # the optimiser family is fixed at 'cfl' (per-voxel Adam normalisation roughens the
            # velocity); optimizer_lr / max_step_norm apply to the Adam family only
        ],
    ),
    "syn": MethodSpec(
        name="syn", model="sobolev", function="syntx.syn",
        defaults=_signature_defaults(
            "syntx.syn.registration",
            ["grad_step", "flow_sigma", "total_sigma", "sobolev_alpha", "fast_smooth",
             "in_loop_inv_steps", "optimizer"],
            hidden={"regularizer": "sobolev", "fast_smooth": False,
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


def _sq(v):
    """``v ** 2`` rounded to 9 decimals (None stays None)."""
    return None if v is None else round(float(v) ** 2, 9)


def _as_bool(v):
    """``bool(v)``."""
    return bool(v)


# Canonical-default declarations (where each method's defaults live, how to probe them).
# tests/test_canonical_parameters.py and syntx.benchmark.codify are driven by these.
_CANONICAL = {
    "greedy": dict(
        function_file="src/syntx/greedy.py", function_name="greedy_registration",
        config_block="greedy_config", run_config_block="greedy_config",
        config_keys={"grad_step": "learning_rate", "flow_sigma": "flow_sigma",
                     "total_sigma": "total_sigma", "optimizer": "optimizer",
                     "regadam_sigma": "regadam_sigma", "similarity_metric": "similarity_metric",
                     "lncc_radius": "lncc_radius", "reg_iterations": "reg_iterations"},
        resolved={"learning_rate": ("attr", "learning_rate", None),
                  "flow_sigma": ("attr", "flow_sigma", None),
                  "total_sigma": ("attr", "total_sigma", None),
                  "optimizer": ("attr", "optimizer", None),
                  "regadam_sigma": ("attr", "regadam_sigma", None),
                  "lncc_radius": ("attr", "lncc_radius", None),
                  "similarity_metric": ("attr", "similarity_metric", None)},
        probe_kwargs={"initial_transform": False},
        tests=("tests/test_greedy.py",),
    ),
    "syn": dict(
        function_file="src/syntx/syn.py", function_name="registration",
        config_block="syn_config", run_config_block="syn_config",
        config_keys={"grad_step": "grad_step", "fluid_sigma": "flow_sigma",
                     "elastic_sigma": "total_sigma", "lncc_radius": "syn_sampling",
                     "in_loop_inv_steps": "in_loop_inv_steps", "syn_metric": "syn_metric",
                     "syn_regularizer": "regularizer", "kernel_type": "kernel_type",
                     "sobolev_alpha": "sobolev_alpha", "syn_fast_smooth": "fast_smooth",
                     "syn_use_analytical_gradients": "use_analytical_gradients",
                     "syn_inverse_method": "inverse_method", "syn_formulation": "formulation",
                     "reg_iterations": "reg_iterations"},
        resolved={"grad_step": ("fit", "cfl_voxels", None),
                  "flow_sigma": ("attr", "fluid_sigma", _sq),      # model stores sigma
                  "total_sigma": ("attr", "elastic_sigma", _sq),
                  "syn_sampling": ("fit", "lncc_radius", None),
                  "in_loop_inv_steps": ("attr", "in_loop_inv_steps", None),
                  "syn_metric": ("fit", "similarity_metric", None),
                  "regularizer": ("fit", "regularizer", None),
                  "sobolev_alpha": ("fit", "sobolev_alpha", None),
                  "fast_smooth": ("fit", "fast_smooth", _as_bool),
                  "use_analytical_gradients": ("fit", "use_analytical_gradients", _as_bool),
                  "inverse_method": ("attr", "inverse_method", None),
                  "formulation": ("attr", "formulation", None),
                  "kernel_type": ("attr", "kernel_type", None),
                  "optimizer": ("fit", "optimizer_type", None)},
        probe_kwargs={"initial_transform": "identity"},
        equivalent={"kernel_type": {"sobolev": "bessel"}},  # same filter branch
        tests=("tests/test_syn.py",),
    ),
    "tvf": dict(
        function_file="src/syntx/tvf.py", function_name="tvf_registration",
        config_block="tvf_config", run_config_block="tvf_config",
        config_keys={k: k for k in ("regularizer", "alpha", "optimizer", "grad_step", "cfl_momentum",
                                    "energy_weight", "n_time_steps", "multipoint_loss",
                                    "fast_smooth", "syn_metric", "syn_sampling", "reg_iterations")},
        resolved={"regularizer": ("fit", "regularizer", None),
                  "alpha": ("fit", "alpha", None),
                  "total_alpha": ("fit", "total_alpha", None),
                  "optimizer": ("fit", "optimizer_type", None),
                  "grad_step": ("fit", "cfl_step", None),
                  "cfl_momentum": ("fit", "cfl_momentum", None),
                  "energy_weight": ("fit", "energy_weight", None),
                  "multipoint_loss": ("fit", "multipoint_loss", None),
                  "fast_smooth": ("fit", "fast_smooth", None),
                  "syn_metric": ("fit", "similarity_metric", None),
                  "syn_sampling": ("fit", "lncc_radius", None),
                  "cfl_max": ("fit", "cfl_max", None),
                  "n_time_steps": ("attr", "n_time_steps", None)},
        constant_defaults={"alpha": ("src/syntx/tvf.py", "TVF_DEFAULT_ALPHA", 3)},
        probe_kwargs={"initial_transform": "identity"},
        tests=("tests/test_tvf_interface.py", "tests/test_parameter_sensitivity.py"),
    ),
    "syngs": dict(
        function_file="src/syntx/syngs.py", function_name="syngs_registration",
        config_block="syngs_config", run_config_block="syngs_config",
        config_keys={"grad_step": "grad_step",
                     "total_sigma": "total_sigma", "alpha": "alpha", "regularizer": "regularizer",
                     "optimizer": "optimizer", "optimizer_lr": "optimizer_lr",
                     "max_step_norm": "max_step_norm", "syn_metric": "syn_metric",
                     "n_steps": "n_steps", "bootstrap_mode": "bootstrap_mode",
                     "reg_iterations": "reg_iterations"},
        resolved={"grad_step": ("fit", "cfl_step", None),
                  "total_sigma": ("attr", "elastic_sigma", None),
                  "alpha": ("attr", "alpha", None),
                  "regularizer": ("attr", "regularizer", None),
                  "optimizer": ("fit", "optimizer_type", None),
                  "optimizer_lr": ("fit", "lr", None),
                  "max_step_norm": ("fit", "max_step_norm", None),
                  "syn_metric": ("fit", "similarity_metric", None),
                  "n_steps": ("attr", "n_steps", None),
                  "bootstrap_mode": ("attr", "bootstrap_mode", None)},
        constant_defaults={"alpha": ("src/syntx/syngs.py", "SYNGS_DEFAULT_ALPHA", 3)},
        probe_kwargs={"initial_transform": "identity"},  # no affine: fast, CPU-only probe
        tests=("tests/test_syngs_coverage.py",),
    ),
}
for _name, _decl in _CANONICAL.items():
    for _k, _v in _decl.items():
        setattr(METHODS[_name], _k, _v)

CANONICAL_POINTER = "docs/provenance/canonical.json"  # method -> canonical best_parameters record


def probe_effective_defaults(spec: MethodSpec) -> Dict[str, Any]:
    """The defaults ``spec``'s registration function *actually* runs with, captured at the
    model's fit() on a tiny image (so body-resolved hidden defaults are included).

    Runs ``syntx.<function>`` once on a random 20^3 image pair on CPU with
    ``reg_iterations=[1, 1, 1]`` plus ``spec.probe_kwargs``, and reads each ``spec.resolved``
    entry from the captured fit() keywords or model attributes (applying its transform).

    Returns
    -------
    dict
        Parameter -> effective value.

    Raises
    ------
    KeyError
        A declared key was not captured.
    """
    import numpy as np
    import ants
    import syntx
    from syntx.provenance import build_manifest, capture_registration_calls, resolved_parameters
    arr = np.random.default_rng(0).random((20, 20, 20)).astype(np.float32) * 100
    fixed, moving = ants.from_numpy(arr), ants.from_numpy(np.roll(arr, 1, 0))
    kw = {"device": "cpu", "verbose": False, "reg_iterations": [1, 1, 1], **spec.probe_kwargs}
    with capture_registration_calls() as cap:
        getattr(syntx, spec.function.split(".", 1)[1])(fixed=fixed, moving=moving, **kw)
    p = resolved_parameters(build_manifest(calls=cap.calls, include_diff=False),
                            function_prefix=spec.function)
    out = {}
    for param, (src, key, tf) in spec.resolved.items():
        pool = p["fit_kwargs"] if src == "fit" else p["model_attributes"]
        if key not in pool:
            raise KeyError(f"{spec.name}: {src} {key!r} not captured (for {param!r})")
        out[param] = tf(pool[key]) if tf else pool[key]
    return out


def expected_defaults(spec: MethodSpec, probed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Expected default per parameter: probed value, else the function's signature default,
    else (where the signature default is None or absent) ``spec.defaults()``'s value."""
    import importlib
    import inspect
    mod = importlib.import_module(f"syntx.{os.path.splitext(os.path.basename(spec.function_file))[0]}")
    sig = inspect.signature(getattr(mod, spec.function_name)).parameters
    out = {n: p.default for n, p in sig.items() if p.default is not inspect.Parameter.empty}
    out.update({k: v for k, v in spec.defaults().items() if out.get(k) is None})
    out.update(probed or {})
    return out


def canonical_mismatches(spec: MethodSpec, values: Dict[str, Any], expected: Dict[str, Any],
                         keys_are_config: bool = True) -> Dict[str, Any]:
    """{key: (value, expected)} for every entry of ``values`` that disagrees with ``expected``.

    With ``keys_are_config`` the keys are config keys mapped through ``spec.config_keys``;
    unmapped keys or parameters without an expected value are reported with a '<...>'
    marker. Floats compare with an absolute tolerance of 1e-9, sequences element-wise, and
    ``spec.equivalent`` values are treated as equal.
    """
    bad = {}
    for k, v in values.items():
        param = spec.config_keys.get(k) if keys_are_config else k
        if param is None:
            bad[k] = (v, "<unmapped key>")
            continue
        if param not in expected:
            bad[k] = (v, "<no known default>")
            continue
        eq = spec.equivalent.get(param, {})
        canon = lambda x: eq.get(x, x) if isinstance(x, (str, int, float, bool)) or x is None else x
        a, b = canon(v), canon(expected[param])
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and abs(float(a) - float(b)) < 1e-9
        else:
            same = (list(a) == list(b)) if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) else a == b
        if not same:
            bad[k] = (v, expected[param])
    return bad


# ----------------------------------------------------------------------------------------
# Constraints / scoring
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class Criteria:
    """Per-pair feasibility constraints, judged *relative to the baseline* (the current
    defaults), since even the defaults fold slightly / exceed 1 mm inverse error on some pairs.
    Used by ``pair_violations``; saved to ``<out_dir>/criteria.json``. On the command line
    only ``jac_min_rel`` can be set (``--jac-min-rel``); the other fields keep these defaults.

    Attributes
    ----------
    folding_abs : float, default 0.0005
        Folding (in %, i.e. 0.0005 % of masked voxels) always allowed ...
    folding_rel : float, default 1.0
        ... or up to baseline folding x this.
    inv_interior_abs : float, default 1.0
        Interior max inverse-identity error (mm) always allowed ...
    inv_interior_rel : float, default 1.1
        ... or up to baseline x this.
    positive_jac_if_baseline : bool, default True
        Where the baseline's minimum Jacobian is > 0, require it to stay > 0.
    inverse_cap_only_if_baseline_fold_free : bool, default True
        Apply the inverse cap only on pairs where the baseline does not fold.
    max_pair_drop : float, default 0.001
        Maximum Dice drop below the baseline on any pair.
    jac_min_rel : float or None, default None
        Require minimum Jacobian >= this x the baseline's (only where the baseline's is > 0).
        The minimum is the record's 'syntx_min_jac': the ``syntx.liouville_determinant`` value
        for syntx syn / tvf / syngs / greedy results, else the finite-difference determinant
        of the exported warp. None disables it.
    min_gain : float, default 0.0005
        Lower bound of the acceptance margin (mean Dice).
    noise_k : float, default 3.0
        Margin = max(min_gain, noise_k x baseline rep-to-rep noise).
    """
    folding_abs: float = 0.0005        # folding % always allowed up to this ...
    folding_rel: float = 1.0           # ... or up to baseline folding * folding_rel
    inv_interior_abs: float = 1.0      # interior max inverse error (mm) allowed up to this ...
    inv_interior_rel: float = 1.1      # ... or up to baseline * inv_interior_rel
    positive_jac_if_baseline: bool = True
    # The inverse cap is relative to the baseline's inverse error, which is not a meaningful
    # reference where the baseline itself folds: apply it only on fold-free baseline pairs.
    inverse_cap_only_if_baseline_fold_free: bool = True
    max_pair_drop: float = 0.001       # no pair's Dice may fall more than this below baseline
    # true (syntx.liouville_determinant) minimum Jacobian determinant must stay >= this fraction
    # of the baseline's -- bounds local compression; None = off
    jac_min_rel: Optional[float] = None
    min_gain: float = 0.0005
    noise_k: float = 3.0


METRIC_KEYS = {
    "dice_sym": "syntx_dice_sym", "dice_fixed": "syntx_dice_fixed", "dice_moving": "syntx_dice_moving",
    "folding_pct": "syntx_fold", "jac_min": "syntx_min_jac", "jac_max": "syntx_max_jac",
    "inv_mean_mm": "syntx_inv_mean", "inv_p95_mm": "syntx_inv_p95", "inv_max_mm": "syntx_inv_max",
    "inv_interior_mean_mm": "syntx_inv_interior_mean", "inv_interior_max_mm": "syntx_inv_interior_max",
    "time_s": "syntx_time",
    "fd_folding_pct": "syntx_fd_fold", "fd_jac_min": "syntx_fd_min_jac",
}


def _num(v):
    """``float(v)``, or NaN if it cannot be converted."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _nan(v) -> bool:
    """True only for a float NaN."""
    return isinstance(v, float) and math.isnan(v)


def pair_violations(m: Dict[str, float], base: Dict[str, float], c: Criteria,
                    has_inverse: bool) -> List[str]:
    """Constraint violations of one pair's metrics ``m`` against the baseline ``base``.

    Checks, in order: Dice drop > ``max_pair_drop``; folding % above
    max(``folding_abs``, baseline x ``folding_rel``); minimum Jacobian not > 0 where the
    baseline's is > 0 (``positive_jac_if_baseline``); minimum Jacobian below
    ``jac_min_rel`` x baseline; and, if ``has_inverse`` and the baseline's interior max
    inverse error is not NaN (and, by default, the baseline does not fold), interior max
    inverse error NaN or above max(``inv_interior_abs``, baseline x ``inv_interior_rel``).
    A NaN Dice or folding value of ``m`` does not trigger the Dice or folding checks.

    Returns
    -------
    list of str
        Human-readable violations; empty if feasible.
    """
    out = []
    if m["dice_sym"] < base["dice_sym"] - c.max_pair_drop:
        out.append(f"dice drop {base['dice_sym'] - m['dice_sym']:.4f} > {c.max_pair_drop}")
    fold_cap = max(c.folding_abs, base["folding_pct"] * c.folding_rel)
    if m["folding_pct"] > fold_cap + 1e-12:
        out.append(f"folding {m['folding_pct']:.4f}% > {fold_cap:.4f}%")
    if c.positive_jac_if_baseline and base["jac_min"] > 0 and not m["jac_min"] > 0:
        out.append("jacobian min reached 0 (baseline > 0)")
    if c.jac_min_rel is not None and base["jac_min"] > 0 and m["jac_min"] < c.jac_min_rel * base["jac_min"]:
        out.append(f"jacobian min {m['jac_min']:.4f} < {c.jac_min_rel} x baseline {base['jac_min']:.4f}")
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
    """sha256 hex digest of ``obj`` as sorted-key JSON (non-JSON values via ``str``)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _affine_sha(pair: int) -> Optional[str]:
    """sha256 of the Mindboggle canonical affine file of ``pair`` (``AFFINE_CACHE``), or None
    if it does not exist."""
    p = AFFINE_CACHE.format(pair=pair)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# 2-D tuning pairs (ANTs example slices; 3-class Otsu labels, as benchmark_data('2d')).
TWO_D_PAIRS = {0: ("r16", "r64"), 1: ("r27", "r85"), 2: ("r30", "r62")}
TWO_D_AFFINE_CACHE = "results/canonical_affines_2d/pair2d_{pair}_affine.mat"


def twod_evaluator(spec: MethodSpec, device: str = "cpu"):
    """Evaluator on the 2-D pairs ``TWO_D_PAIRS`` (ANTs example slices; fast, and
    deterministic on CPU).

    Same metric keys as the Mindboggle evaluator: Dice of 3-class Otsu labels (classes 1-3)
    via ``compute_bidirectional_dice``; Jacobian folding / min / max from
    ``flow_jacobian_metrics`` (the method's own determinant) when available, else the
    finite-difference determinant of the forward warp; inverse-identity error from
    ``res['inverse_identity_errors']['phi_1']`` (all voxels + 5-voxel-eroded interior; NaN
    if the method has no inverse or no 'phi_1' entry). The call is
    ``syntx.<method>(fixed, moving, initial_transform=affine, device=device, verbose=False,
    **overrides)`` -- the method's own defaults plus the overrides -- on the raw
    (not intensity-normalised) images. 'time_s' is the deformable call only.

    The affine is ``syntx.robust_affine(fi, mi)`` with its default settings (not forced to
    CPU), computed once per pair and cached at ``TWO_D_AFFINE_CACHE`` (relative to the
    working directory), so it is held constant across evaluations and tunes.

    Parameters
    ----------
    spec : MethodSpec
        Method to run (``spec.function``).
    device : str, default 'cpu'
        Device of the registration calls.

    Returns
    -------
    callable
        ``run(pair, overrides, report_dir=None) -> (metrics, None)`` (no record, so no
        provenance or canonical check; ``report_dir`` is ignored), with attribute
        ``run.affine_sha(pair)``. A result without a ``.nii.gz`` forward transform raises
        StopIteration.
    """
    import ants
    import importlib
    import shutil
    import syntx
    from syntx.benchmark.evaluate import _inverse_error_stats
    from syntx.deformation_metrics import (compute_bidirectional_dice, compute_jacobian_metrics,
                                           flow_jacobian_metrics)

    fn = getattr(syntx, spec.function.split(".")[-1])
    cache = {}

    def load(pair):
        if pair not in cache:
            f_key, m_key = TWO_D_PAIRS[pair]
            fi = ants.image_read(ants.get_ants_data(f_key))
            mi = ants.image_read(ants.get_ants_data(m_key))
            fl = ants.threshold_image(fi, "Otsu", 3)
            ml = ants.threshold_image(mi, "Otsu", 3)
            aff = TWO_D_AFFINE_CACHE.format(pair=pair)
            if not os.path.exists(aff):
                os.makedirs(os.path.dirname(aff), exist_ok=True)
                shutil.copyfile(syntx.robust_affine(fi, mi)["fwdtransforms"][0], aff)
            cache[pair] = (fi, mi, fl, ml, aff)
        return cache[pair]

    def run(pair: int, overrides: Dict[str, Any], report_dir: Optional[str] = None):
        fi, mi, fl, ml, aff = load(pair)
        t0 = time.time()
        res = fn(fixed=fi, moving=mi, initial_transform=aff, device=device, verbose=False, **overrides)
        t = time.time() - t0
        d_fix, d_mov, d_sym = compute_bidirectional_dice(fl, ml, fi, mi, res["fwdtransforms"],
                                                         res["invtransforms"], res.get("whichtoinvert_inv"))
        warp = next(x for x in res["fwdtransforms"] if x.endswith(".nii.gz"))
        jac_fd = compute_jacobian_metrics(fi, warp)
        jac = flow_jacobian_metrics(fi, res) or jac_fd        # exact flow determinant if available
        inv = {k: float("nan") for k in ("mean", "p95", "max", "interior_mean", "interior_max")}
        if spec.has_inverse:
            err = (res.get("inverse_identity_errors") or {}).get("phi_1", {})
            inv = _inverse_error_stats(err, fi)
        metrics = {"dice_sym": float(d_sym), "dice_fixed": float(d_fix), "dice_moving": float(d_mov),
                   "folding_pct": float(jac["folding_pct"]), "jac_min": float(jac["min"]),
                   "jac_max": float(jac["max"]), "inv_mean_mm": inv["mean"], "inv_p95_mm": inv["p95"],
                   "inv_max_mm": inv["max"], "inv_interior_mean_mm": inv["interior_mean"],
                   "inv_interior_max_mm": inv["interior_max"], "time_s": t,
                   "fd_folding_pct": float(jac_fd["folding_pct"]), "fd_jac_min": float(jac_fd["min"])}
        return metrics, None

    def affine_sha(pair):
        p = TWO_D_AFFINE_CACHE.format(pair=pair)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    run.affine_sha = affine_sha
    return run


def mindboggle_evaluator(spec: MethodSpec, generate_report: bool = False, **fixed_kwargs):
    """Default evaluator: ``evaluate_mindboggle_pair(pair, spec.model, **overrides)``.

    The Tuner passes only overrides that differ from the method's defaults, so the baseline
    is the standard benchmark path (``config=None``: the method's own defaults).

    Parameters
    ----------
    spec : MethodSpec
        Method to run (``spec.model``).
    generate_report : bool, default False
        Passed to ``evaluate_mindboggle_pair``.
    **fixed_kwargs
        Extra keywords passed on every call (they would show up as explicit parameters in
        ``check_canonical_call`` unless they are plumbing).

    Returns
    -------
    callable
        ``run(pair, overrides, report_dir=None) -> (metrics, record)``: ``metrics`` maps the
        ``METRIC_KEYS`` names to floats taken from the record (NaN if missing; inverse
        metrics forced to NaN when ``spec.has_inverse`` is False); ``record`` is the full
        evaluation record including 'provenance'.
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
    """Raise NotCanonicalError if the evaluator passed tuning keywords beyond ``overrides``.

    Reads the explicit keywords of the recorded ``spec.function`` call from the record's
    provenance manifest; anything not in ``spec.evaluator_plumbing`` or ``overrides`` is an
    error. Also raised when the record has no manifest or no such call.
    """
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
    """Drop ``__version__`` lines from the package ``__init__.py``; other files unchanged."""
    if rel == "__init__.py":  # version bumps do not change registration
        data = b"\n".join(l for l in data.split(b"\n") if not l.lstrip().startswith(b"__version__"))
    return data


def registration_code_fingerprint(commit: Optional[str] = None, pkg_dir: Optional[str] = None) -> Optional[str]:
    """sha256 over the syntx package sources that can affect registration results.

    All files of the package directory (not only ``.py``) are hashed by relative path and
    content, excluding ``TUNER_ONLY_FILES``, ``__pycache__`` and the ``__version__`` line of
    ``__init__.py``. ``commit=None`` reads the working tree (tracked files + untracked,
    non-ignored files); otherwise the files at ``commit`` (results for a commit are memoised
    in-process).

    Parameters
    ----------
    commit : str, optional
        Git revision; None for the working tree.
    pkg_dir : str, optional
        Package directory; default that of the imported ``syntx``.

    Returns
    -------
    str or None
        Hex digest; None if the package is not in a git checkout, git fails, or the commit
        has no package files.
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
    """Append-only JSON-lines cache of evaluation rows, indexed by ``row['key']``.

    Loaded fully at construction (a later line with the same key replaces an earlier one);
    ``put`` updates the index and appends one line to ``path``.
    """
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
        """Row for ``key`` or None."""
        return self.rows.get(key)

    def put(self, row):
        """Store ``row`` and append it to the file."""
        self.rows[row["key"]] = row
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


# ----------------------------------------------------------------------------------------
# Tuner
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    """One evaluated configuration: ``overrides`` of the defaults, the stage that first
    created it, and ``per_pair`` = {pair: [metrics dict per repeat]}."""
    overrides: Dict[str, Any]
    stage: str
    per_pair: Dict[int, List[Dict[str, float]]] = dataclasses.field(default_factory=dict)

    @property
    def label(self) -> str:
        """'k=v, ...' of the overrides (sorted), or 'defaults'."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.overrides.items())) or "defaults"

    def mean_metric(self, pair: int, k: str) -> float:
        """Mean of metric ``k`` over the repeats of ``pair``, skipping NaN (NaN if none)."""
        vals = [r[k] for r in self.per_pair[pair] if not _nan(r.get(k, float("nan")))]
        return sum(vals) / len(vals) if vals else float("nan")

    def summary(self, pairs) -> Dict[str, float]:
        """{pair: {metric: repeat mean}} for every ``METRIC_KEYS`` metric."""
        return {p: {k: self.mean_metric(p, k) for k in METRIC_KEYS} for p in pairs}


class Tuner:
    """Runs one automated tune (see the module docstring for the search procedure).

    Parameters
    ----------
    method : str or MethodSpec
        A ``METHODS`` key ('greedy', 'syn', 'syngs', 'tvf') or a spec.
    pairs : sequence of int, default ``DEFAULT_PAIRS`` (77, 44, 0)
        Pairs passed to the evaluator.
    space : list of Param, optional
        Search space; default ``spec.space``. Parameters in ``fixed_parameters`` are removed.
    criteria : Criteria, optional
        Feasibility constraints; default ``Criteria()``.
    evaluator : callable, optional
        ``evaluator(pair, overrides) -> (metrics, record or None)``, optionally with an
        ``affine_sha(pair)`` attribute; default ``mindboggle_evaluator(spec)``.
    out_dir : str, optional
        Output directory; default ``results/tune_<method>_<YYYYMMDD>`` (relative to the working
        directory, so tunes of the same method on the same day share it and its cache).
    max_evals : int, default 300
        Budget of new (uncached) single-pair evaluations.
    max_hours : float, optional
        Wall-clock budget. When a budget runs out, the search stops and the result is
        written with what has been evaluated.
    refine_rounds : int, default 6
        Maximum number of refine rounds.
    top_k : int, default 3
        Number of improving moves combined per round (and of gainers / clean moves in
        compensating pairs).
    allow_dirty : bool, default False
        Allow an uncommitted checkout (otherwise RuntimeError at construction).
    check_canonical : bool, default True
        Run ``check_canonical_call`` on every new record.
    code_fingerprint : dict, optional
        ``{'commit', 'diff_sha256', 'registration_code'}``; default from
        ``syntx.provenance.code_state`` (the clean-checkout check is skipped when given).
    log : callable, optional
        Message sink; default ``print``.
    fixed_parameters : dict, optional
        Parameters that are not searched; default ``DEFAULT_FIXED_PARAMETERS``
        (reg_iterations). Their values are not passed to the evaluator: the method's own
        default is used. Overrides touching them raise ValueError.
    start : dict, optional
        Starting point: evaluated first (stage 'start') and used as the centre of the first
        screen. Must only name search-space parameters (ValueError in ``run``). Gains,
        feasibility and the incumbent best stay relative to the defaults.

    Side effects at construction: creates ``<out_dir>/runs`` and writes
    ``<out_dir>/criteria.json``.
    """
    def __init__(self, method: str, pairs: Sequence[int] = DEFAULT_PAIRS,
                 space: Optional[List[Param]] = None, criteria: Criteria = None,
                 evaluator: Optional[Callable] = None, out_dir: Optional[str] = None,
                 max_evals: int = 300, max_hours: Optional[float] = None,
                 refine_rounds: int = 6, top_k: int = 3, allow_dirty: bool = False,
                 check_canonical: bool = True, code_fingerprint: Optional[Dict] = None,
                 log: Optional[Callable[[str], None]] = None,
                 fixed_parameters: Optional[Dict[str, Any]] = None,
                 start: Optional[Dict[str, Any]] = None):
        self.spec = METHODS[method] if isinstance(method, str) else method
        # Optional starting point: overrides evaluated first and used as the centre of the
        # screen. Gains / feasibility are still measured against the true defaults.
        self.start = dict(start or {})
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
        with open(os.path.join(self.out_dir, "criteria.json"), "w") as f:   # read by --table
            json.dump(dataclasses.asdict(self.criteria), f, indent=2)
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
        """False once ``max_evals`` new evaluations were made or ``max_hours`` elapsed."""
        if self.n_new >= self.max_evals:
            return False
        if self.max_hours is not None and (time.time() - self.t0) / 3600.0 > self.max_hours:
            return False
        return True

    def evaluate(self, overrides: Dict[str, Any], stage: str, rep: int = 0) -> Optional[Config]:
        """Evaluate ``overrides`` (relative to the defaults) on every pair, repeat ``rep``.

        Overrides equal to the defaults are dropped first. For each pair whose repeat ``rep``
        is not yet stored in the configuration, a cached row (``_cached``) is used, or the
        evaluator is run: the record is checked (``_check_record``) and saved as
        ``runs/<key[:16]>.json``, and the row is appended to ``evaluations.jsonl``. The
        pair's affine sha is checked and ``live.md`` / ``live.csv`` are rewritten.

        Returns
        -------
        Config or None
            The configuration, or None if the budget ran out before all pairs were done
            (pairs already done stay stored).

        Raises
        ------
        ValueError
            ``overrides`` touch a fixed parameter.
        RuntimeError, NotCanonicalError
            From the record / affine checks.
        """
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
                       "affine_sha256": getattr(self.evaluator, "affine_sha", _affine_sha)(pair),
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
        """Cache key: hash of method, pair, overrides, rep and the registration-code identity."""
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
        """RuntimeError if the record's provenance says code changed during the run; then
        ``check_canonical_call`` when enabled."""
        man = record.get("provenance") or {}
        if man.get("changed_during_run"):
            raise RuntimeError(f"code changed on disk during the pair-{pair} run; aborting")
        if self.check_canonical:
            check_canonical_call(self.spec, record, overrides)

    def _check_affine(self, pair, sha):
        """RuntimeError if ``sha`` differs from the first affine sha seen for ``pair`` (None:
        no check)."""
        if sha is None:
            return
        prev = self.affine.setdefault(pair, sha)
        if prev != sha:
            raise RuntimeError(f"pair {pair}: affine changed ({prev[:12]} -> {sha[:12]}); "
                               f"the affine must be held constant")

    # -- scoring -----------------------------------------------------------------------------
    def score(self, cfg: Config) -> Tuple[float, bool, Dict[int, List[str]]]:
        """(gain, feasible, violations) of ``cfg`` against the baseline.

        gain = mean over pairs of (repeat-mean Dice - baseline repeat-mean Dice); feasible =
        no ``pair_violations`` on any pair; violations = {pair: [messages]}.
        """
        base = self.baseline.summary(self.pairs)
        s = cfg.summary(self.pairs)
        gain = sum(s[p]["dice_sym"] - base[p]["dice_sym"] for p in self.pairs) / len(self.pairs)
        viol = {p: pair_violations(s[p], base[p], self.criteria, self.spec.has_inverse)
                for p in self.pairs}
        return gain, not any(viol.values()), viol

    def _complete(self, cfg):
        """True if ``cfg`` has at least one run on every pair."""
        return cfg is not None and all(cfg.per_pair.get(p) for p in self.pairs)

    def _value(self, overrides, name):
        """Value of ``name`` in ``overrides``, else its default."""
        return overrides.get(name, self.defaults.get(name))

    def _full(self, overrides):
        """Defaults updated with ``overrides``."""
        full = dict(self.defaults)
        full.update(overrides)
        return full

    # -- search ------------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Run baseline, screen and refine (module docstring) and return ``result(...)``.

        Raises
        ------
        RuntimeError
            The budget ran out before both baseline repeats were measured.
        ValueError
            ``start`` names parameters outside the search space.
        """
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

        if self.start:
            bad = sorted(set(self.start) - {p.name for p in self.space})
            if bad:
                raise ValueError(f"start parameters {bad} are not in the search space")
            self.log(f"starting point: {self.start}")
            self.evaluate(dict(self.start), "start")
        screen(dict(self.start), "screen")
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
        """Complete configurations sorted feasible first, then by gain (descending).

        Each row: 'overrides', 'label', 'stage', 'gain', 'feasible', 'violations',
        'n_reps' (minimum over pairs), 'per_pair' (``Config.summary``), 'mean_dice'.
        """
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
        """Assemble the result, write ``result.json`` and ``report.md`` and return it.

        Keys: 'method', 'pairs', 'defaults', 'out_dir', 'fixed_parameters', 'criteria',
        'noise', 'margin', 'code', 'affine_sha256' ({pair: sha}), 'baseline' (per-pair
        summary), 'winner' ('overrides' -- {} if nothing was confirmed --, 'gain',
        'parameters' (defaults + overrides), 'per_pair'), 'improved', 'n_configs',
        'n_new_evaluations', 'ranking'.
        """
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
        """Provenance manifests of the winner's rep-0 runs, one per pair (read from
        ``runs/``; fails for evaluators that return no record)."""
        best = result["winner"]["overrides"]
        out = []
        for pair in self.pairs:
            row = self._cached(pair, best, 0)
            with open(os.path.join(self.out_dir, row["record_file"])) as f:
                out.append(json.load(f)["provenance"])
        return out


def _mm(v) -> str:
    """'NaN' or the value with 2 decimals."""
    return "NaN" if _nan(_num(v)) else f"{_num(v):.2f}"


def accumulated_table(out_dir: str, include_fixed: bool = False) -> str:
    """Markdown table of every configuration evaluated so far, read from the tune's
    ``evaluations.jsonl`` (works while a tune is running, whatever code it runs).

    One row per configuration (all rows of the file are grouped by overrides, whatever code
    state produced them): runs (minimum over pairs), mean Dice over pairs, gain vs the
    defaults row, feasibility (``pair_violations`` with ``<out_dir>/criteria.json`` if
    present, else default ``Criteria``), and per pair Dice / folding % / Jacobian min /
    interior max inverse / global max inverse (mm) / time (s), averaged over repeats. Rows
    sorted: complete first, then feasible, then mean Dice. '...' marks values not yet
    available.

    Parameters
    ----------
    out_dir : str
        The tune's output directory.
    include_fixed : bool, default False
        Also show runs whose overrides touch a ``DEFAULT_FIXED_PARAMETERS`` key (hidden by
        default; their number is noted in the title).

    Returns
    -------
    str
        Markdown text ('no evaluations yet' when empty).
    """
    rows = []
    with open(os.path.join(out_dir, "evaluations.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    crit_path = os.path.join(out_dir, "criteria.json")
    criteria = Criteria(**json.load(open(crit_path))) if os.path.exists(crit_path) else Criteria()
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
            viol = [v for p in pairs for v in pair_violations(per[p], bper[p], criteria, has_inverse)]
            feas = "yes" if not viol else "no"
        table.append((mean, label, g["stage"], runs, gain, per, complete, feas))
    table.sort(key=lambda t: (t[6], t[7] == "yes", t[0] if not _nan(t[0]) else -1), reverse=True)
    head = ("| # | configuration | stage | runs | mean Dice | gain | feasible | " +
            " | ".join(f"pair {p}: Dice / fold% / Jmin / inv int max / inv max / s" for p in pairs) + " |")
    note = f"; {n_fixed} runs touching fixed parameters hidden (include_fixed=True shows them)" if n_fixed else ""
    out = [f"# Accumulated tuning results: `{out_dir}` ({len(rows)} evaluations, {len(table)} configurations{note})",
           f"(feasible = per-pair constraints vs the defaults, criteria {'criteria.json' if os.path.exists(crit_path) else 'default'}; rows sorted feasible first, then mean Dice)",
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


def winner_manifests_from_dir(out_dir: str, method: str, overrides: Dict[str, Any],
                              pairs: Sequence[int]) -> List[Dict[str, Any]]:
    """Provenance manifests of the rep-0 runs of ``overrides`` in a tune's output directory,
    one per pair in ``pairs`` order (first matching row of ``evaluations.jsonl`` per pair).

    Raises
    ------
    KeyError
        A pair has no recorded rep-0 run of ``overrides``.
    """
    want = json.dumps(overrides, sort_keys=True, default=str)
    rows = {}
    with open(os.path.join(out_dir, "evaluations.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("method") == method and r.get("rep") == 0 and r.get("record_file")
                    and json.dumps(r.get("overrides"), sort_keys=True, default=str) == want):
                rows.setdefault(int(r["pair"]), r)
    missing = [p for p in pairs if int(p) not in rows]
    if missing:
        raise KeyError(f"no recorded runs of {overrides} for pairs {missing} in {out_dir}")
    out = []
    for p in pairs:
        with open(os.path.join(out_dir, rows[int(p)]["record_file"])) as f:
            out.append(json.load(f)["provenance"])
    return out


def render_report(res: Dict[str, Any]) -> str:
    """Markdown report of a ``Tuner.result`` dict: settings, winner, ranked table of all
    configurations, and the violations of the infeasible ones."""
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

    Parameters
    ----------
    method : str
        A ``METHODS`` key.
    pairs : sequence of int, default ``DEFAULT_PAIRS``
        Evaluation pairs.
    record : bool, default False
        With a confirmed winner, add a record to ``best_parameters_path`` via
        ``syntx.provenance.record_result``: key ``tuned_<method>_<YYYY_MM_DD>`` under
        ``syntx.<method>``, holding the winner's parameters, gain, per-pair and baseline
        metrics, criteria, noise, margin and the report path; provenance from the winner's
        rep-0 run manifests (needs an evaluator that returns records).
    best_parameters_path : str, default 'docs/provenance/best_parameters.json'
        Record file.
    **kwargs
        Passed to ``Tuner``.

    Returns
    -------
    dict
        ``Tuner.result``'s dict, plus 'record_key' when a record was written.
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
    """Clear the terminal and print ``out_dir/live.md`` every ``interval`` seconds until
    Ctrl-C (``once``: print a single time). Returns None."""
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
    (examples/pairs.csv, results/...) are unchanged; only the imported code is pinned
    (``PYTHONPATH=<worktree>/src``). Uncommitted changes are therefore not used. The
    worktree is removed afterwards.

    Parameters
    ----------
    argv : sequence of str
        Arguments for ``python -m syntx.benchmark.tune`` (without ``--isolate``).

    Returns
    -------
    int
        The subprocess's exit code.
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
    """Command line for ``python -m syntx.benchmark.tune``.

    Options
    -------
    --method {greedy, syn, syngs, tvf}
        Method to tune (required unless --watch or --table).
    --pairs N [N ...]
        Pairs; default 77 44 0 for the Mindboggle dataset, 0 1 2 for --dataset 2d.
    --dataset {mindboggle, 2d}, default mindboggle
        'mindboggle': ``mindboggle_evaluator`` (``evaluate_mindboggle_pair``). '2d':
        ``twod_evaluator`` on ``TWO_D_PAIRS`` with device 'cpu' (exploratory; cannot be
        combined with --record / --codify). The default --out directory and the cache key do
        not include the dataset, so a 2-D and a Mindboggle tune of the same method on the
        same day share ``evaluations.jsonl``; pass --out to keep them apart.
    --start NAME=VALUE [...]
        Starting point (``Tuner(start=...)``). VALUE is parsed as JSON (e.g. ``0.3``,
        ``true``, ``[100,50,10]``, ``"sobolev"``), falling back to the raw string (so
        ``regularizer=dsti1`` works). Names must be search-space parameters that are not
        fixed; values equal to the defaults are dropped.
    --jac-min-rel X
        Feasibility: minimum Jacobian determinant >= X x the baseline's, per pair
        (``Criteria(jac_min_rel=X)``; all other criteria keep their defaults).
    --unfix NAME [...]
        Remove names from ``DEFAULT_FIXED_PARAMETERS`` (only 'reg_iterations'). It is then
        searched only if the method's space contains it (only greedy's does).
    --out DIR
        Output directory (default ``results/tune_<method>_<YYYYMMDD>``).
    --max-evals N, default 300 / --max-hours H
        Budgets (new evaluations / wall clock).
    --record
        Write a ``best_parameters.json`` record if a winner is confirmed (``tune``).
    --codify
        If a winner is confirmed, call ``syntx.benchmark.codify.codify`` (new branch,
        tests, commit, push).
    --isolate
        Re-run the command from a detached worktree of HEAD (``run_isolated``), so the
        checkout can be edited while the tune runs.
    --watch DIR [--interval S, default 15]
        Print ``DIR/live.md`` periodically.
    --table DIR [--include-fixed]
        Print ``accumulated_table(DIR)`` and exit.

    Returns
    -------
    dict, int or None
        The tune result; the subprocess exit code with --isolate; None for --table /
        --watch.
    """
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
    ap.add_argument("--pairs", type=int, nargs="+", default=None,
                    help=f"pair indices (default: Mindboggle {list(DEFAULT_PAIRS)}; 2-D {sorted(TWO_D_PAIRS)})")
    ap.add_argument("--jac-min-rel", type=float, default=None,
                    help="feasibility: true minimum Jacobian determinant >= this fraction of the baseline's")
    ap.add_argument("--dataset", choices=("mindboggle", "2d"), default="mindboggle",
                    help="'2d': the fast 2-D pairs TWO_D_PAIRS on CPU (exploratory: no --record / --codify)")
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
    ap.add_argument("--start", nargs="*", default=[], metavar="NAME=VALUE",
                    help="starting point (JSON values, e.g. max_step_norm=0.3): evaluated first "
                         "and used as the centre of the screen; gains stay relative to the defaults")
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
    start = {}
    for item in a.start:
        k, _, v = item.partition("=")
        try:
            start[k] = json.loads(v)
        except ValueError:
            start[k] = v
    extra = {}
    if a.jac_min_rel is not None:
        extra["criteria"] = Criteria(jac_min_rel=a.jac_min_rel)
    if a.dataset == "2d":
        if a.record or a.codify:
            ap.error("--dataset 2d is exploratory: --record / --codify need the Mindboggle benchmark")
        extra["evaluator"] = twod_evaluator(METHODS[a.method])
    pairs = a.pairs if a.pairs is not None else (sorted(TWO_D_PAIRS) if a.dataset == "2d" else list(DEFAULT_PAIRS))
    res = tune(a.method, pairs=pairs, record=a.record, out_dir=a.out,
               max_evals=a.max_evals, max_hours=a.max_hours, fixed_parameters=fixed, start=start, **extra)
    if a.codify and res["improved"]:
        from syntx.benchmark.codify import codify
        out = codify(res, out_dir=res["out_dir"])
        print(f"codify: {out}", flush=True)
    return res


if __name__ == "__main__":
    main()
