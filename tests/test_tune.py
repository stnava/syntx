"""
syntx.benchmark.tune: search / refine / validate logic on a synthetic objective with a
known optimum and constraints (fast, deterministic), plus caching, live monitoring and
the canonical-evaluator guard.
"""

import json
import math
import os

import pytest

from syntx.benchmark.tune import (
    Criteria,
    MethodSpec,
    NotCanonicalError,
    Param,
    Tuner,
    check_canonical_call,
    pair_violations,
)

PAIRS = (1, 2, 3)


def _spec():
    return MethodSpec(
        name="synthetic", model="synthetic", function="synthetic.register",
        defaults=lambda: {"a": 1.0, "b": 2.0, "c": "x"},
        space=[Param("a", lo=0.1, hi=4.0), Param("b", lo=0.1, hi=4.0),
               Param("c", kind="categorical", values=("x", "y"))],
    )


def synthetic_evaluator(calls):
    """Dice peaks at a = 1.5; smaller b raises Dice but folds below b = 1.5; c = 'y' adds
    +0.002; a tiny deterministic per-(pair, rep) jitter plays the role of run noise."""
    def run(pair, overrides):
        calls.append((pair, dict(overrides)))
        a = overrides.get("a", 1.0)
        b = overrides.get("b", 2.0)
        c = overrides.get("c", "x")
        rep = sum(1 for p, o in calls if p == pair and o == overrides) - 1
        dice = 0.60 + 0.01 * pair / 10 + 0.02 * (1 - (a - 1.5) ** 2) + 0.004 * (2.0 - b) \
            + (0.002 if c == "y" else 0.0) + 0.00005 * ((pair + rep) % 2)
        folding = 0.0 if b >= 1.5 else 0.01 * (1.5 - b)
        m = {"dice_sym": dice, "dice_fixed": dice, "dice_moving": dice, "folding_pct": folding,
             "jac_min": 0.02 if folding == 0 else 0.0, "jac_max": 20.0,
             "inv_mean_mm": 0.03, "inv_p95_mm": 0.08, "inv_max_mm": 2.0,
             "inv_interior_mean_mm": 0.04, "inv_interior_max_mm": 0.8, "time_s": 1.0}
        return m, None
    return run


def _tuner(tmp_path, calls, **kw):
    return Tuner(_spec(), pairs=PAIRS, evaluator=synthetic_evaluator(calls),
                 out_dir=str(tmp_path / "tune"), check_canonical=False,
                 code_fingerprint={"commit": "0" * 40, "diff_sha256": "x"}, log=lambda s: None, **kw)


def test_tuner_finds_constrained_optimum(tmp_path):
    calls = []
    res = _tuner(tmp_path, calls).run()
    win = res["winner"]["overrides"]
    assert res["improved"]
    assert win.get("c") == "y"                       # combined with the categorical gain
    assert 1.3 <= win.get("a", 1.0) <= 1.7           # bracketed to the interior optimum
    assert win.get("b", 2.0) >= 1.5                  # the folding region is never accepted
    for p in PAIRS:
        assert res["winner"]["per_pair"][p]["folding_pct"] == 0.0
    # every accepted configuration was confirmed by a repeat run
    top = next(r for r in res["ranking"] if r["overrides"] == win)
    assert top["n_reps"] >= 2 and top["feasible"]
    assert res["margin"] >= Criteria().min_gain
    assert os.path.exists(os.path.join(res["code"] and str(tmp_path / "tune"), "report.md"))


def test_tuner_resumes_from_cache_without_new_evaluations(tmp_path):
    calls = []
    first = _tuner(tmp_path, calls).run()
    n_first = len(calls)
    calls2 = []
    t2 = _tuner(tmp_path, calls2)
    second = t2.run()
    assert calls2 == [] and t2.n_new == 0
    assert second["winner"]["overrides"] == first["winner"]["overrides"]
    assert n_first > 0


def test_budget_stops_search_but_reports(tmp_path):
    calls = []
    res = _tuner(tmp_path, calls, max_evals=len(PAIRS) * 4).run()
    assert len(calls) <= len(PAIRS) * 4
    assert "ranking" in res


def test_live_monitoring_files(tmp_path):
    calls = []
    t = _tuner(tmp_path, calls)
    t.run()
    live = open(os.path.join(t.out_dir, "live.md")).read()
    assert "stage: **done**" in live and "best feasible so far" in live
    lines = open(os.path.join(t.out_dir, "live.csv")).read().strip().splitlines()
    assert lines[0].startswith("stage,pair,rep,configuration,dice_sym")
    assert len(lines) - 1 == len(calls)


def test_pair_violations_relative_to_baseline():
    base = {"dice_sym": 0.61, "folding_pct": 0.004, "jac_min": 0.0, "inv_interior_max_mm": 1.4}
    ok = {"dice_sym": 0.6095, "folding_pct": 0.003, "jac_min": 0.0, "inv_interior_max_mm": 1.5}
    assert pair_violations(ok, base, Criteria(), has_inverse=True) == []
    bad = dict(ok, dice_sym=0.605, folding_pct=0.01, inv_interior_max_mm=3.0)
    v = pair_violations(bad, base, Criteria(), has_inverse=True)
    assert len(v) == 3
    # no inverse (greedy): NaN inverse metrics are not a violation
    nan_inv = dict(ok, inv_interior_max_mm=math.nan)
    assert pair_violations(nan_inv, dict(base, inv_interior_max_mm=math.nan), Criteria(),
                           has_inverse=False) == []
    # jacobian min may not drop to 0 where the baseline stays positive
    assert pair_violations(dict(ok, folding_pct=0.0), dict(base, jac_min=0.02, folding_pct=0.0),
                           Criteria(), True) == ["jacobian min reached 0 (baseline > 0)"]


def test_param_candidates_and_bracket():
    p = Param("x", lo=0.1, hi=1.0)
    assert p.candidates(0.4) == [0.2, 0.3, 0.6, 0.8]
    assert p.bracket(0.6, [0.2, 0.3, 0.4, 0.8]) == [0.5, 0.7]
    assert Param("k", kind="int", values=(1, 3)).candidates(2) == [1, 3]
    r = Param("s", requires={"optimizer": ("regadam",)})
    assert not r.active({"optimizer": "adam"}) and r.active({"optimizer": "regadam"})


def test_check_canonical_call_rejects_evaluator_overrides():
    spec = MethodSpec(name="greedy", model="greedy", function="syntx.greedy",
                      defaults=lambda: {}, space=[])
    rec = {"provenance": {"calls": [{"function": "syntx.greedy", "nested_in": None,
                                     "kwargs": {"fixed": 1, "moving": 1, "learning_rate": 0.5},
                                     "resolved": [{"fit_kwargs": {}, "model_attributes": {}}]}]}}
    with pytest.raises(NotCanonicalError, match="learning_rate"):
        check_canonical_call(spec, rec, overrides={})
    check_canonical_call(spec, rec, overrides={"learning_rate": 0.5})  # requested override


def test_reg_iterations_fixed_by_default_but_unfixable(tmp_path):
    """reg_iterations is held at [100, 100, 20] by default for tuning; a caller may unfix it."""
    from syntx.benchmark.tune import DEFAULT_FIXED_PARAMETERS
    from syntx.benchmark.codify import CodifyError, codify
    assert DEFAULT_FIXED_PARAMETERS == {"reg_iterations": [100, 100, 20]}
    spec = _spec()
    spec.defaults = lambda: {"a": 1.0, "b": 2.0, "c": "x", "reg_iterations": [100, 100, 20]}
    spec.space = spec.space + [Param("reg_iterations", kind="list", values=([100, 100, 50],))]
    kw = dict(pairs=PAIRS, check_canonical=False, log=lambda s: None,
              code_fingerprint={"commit": "0" * 40, "diff_sha256": "x"})
    t = Tuner(spec, evaluator=synthetic_evaluator([]), out_dir=str(tmp_path / "t1"), **kw)
    assert "reg_iterations" not in [p.name for p in t.space]
    with pytest.raises(ValueError, match="fixed for this tune"):
        t.evaluate({"reg_iterations": [100, 100, 50]}, "x")
    res = t.run()
    assert res["fixed_parameters"] == {"reg_iterations": [100, 100, 20]}
    with pytest.raises(CodifyError, match="were fixed during this tune"):
        codify(dict(res, winner={"overrides": {"reg_iterations": [100, 100, 50]}, "gain": 1.0}),
               push=False, repo_root=str(tmp_path))
    unfixed = Tuner(spec, evaluator=synthetic_evaluator([]), out_dir=str(tmp_path / "t2"),
                    fixed_parameters={}, **kw)
    assert "reg_iterations" in [p.name for p in unfixed.space]
