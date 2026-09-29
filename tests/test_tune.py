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
    assert len(v) == 2          # baseline folds: the inverse cap does not apply on this pair
    fold_free = dict(base, folding_pct=0.0)
    v = pair_violations(bad, fold_free, Criteria(), has_inverse=True)
    assert len(v) == 3 and any("inverse" in x for x in v)
    # a candidate that removes the baseline's folding is not rejected for its inverse error
    fixes = dict(ok, folding_pct=0.0, inv_interior_max_mm=1.63)
    assert pair_violations(fixes, dict(base, inv_interior_max_mm=1.37), Criteria(), True) == []
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


def test_accumulated_table_reports_both_inverse_maxima(tmp_path):
    from syntx.benchmark.tune import accumulated_table
    t = _tuner(tmp_path, [])
    t.run()
    tbl = accumulated_table(t.out_dir)
    assert "inv int max / inv max" in tbl and "defaults" in tbl
    assert " / 0.80 / 2.00 / " in tbl  # synthetic interior max 0.8, global max 2.0
    live = open(os.path.join(t.out_dir, "live.md")).read()
    assert "inv int max / inv max" in live


def test_compensating_pairs_escape_the_dice_topology_edge(tmp_path):
    """Defaults on the edge: the step 'a' gains Dice but folds, the smoothing 'b' is clean but
    loses Dice; only together are they feasible and better. One-at-a-time moves cannot find it."""
    spec = MethodSpec(name="edge", model="edge", function="edge.register",
                      defaults=lambda: {"a": 1.0, "b": 1.0},
                      space=[Param("a", lo=0.5, hi=2.0, factors=(0.75, 1.5)),
                             Param("b", lo=0.5, hi=2.0, factors=(0.75, 1.5))])
    calls = []

    def run(pair, ov):
        calls.append(pair)
        a, b = ov.get("a", 1.0), ov.get("b", 1.0)
        dice = 0.6 + 0.01 * (a - 1) - 0.004 * (b - 1)
        fold = 0.01 * max(0.0, a - b)
        return ({"dice_sym": dice, "dice_fixed": dice, "dice_moving": dice, "folding_pct": fold,
                 "jac_min": 0.02 if fold == 0 else 0.0, "jac_max": 20.0, "inv_mean_mm": 0.03,
                 "inv_p95_mm": 0.08, "inv_max_mm": 2.0, "inv_interior_mean_mm": 0.04,
                 "inv_interior_max_mm": 0.8, "time_s": 1.0}, None)

    t = Tuner(spec, pairs=PAIRS, evaluator=run, out_dir=str(tmp_path / "edge"), check_canonical=False,
              code_fingerprint={"commit": "0" * 40, "diff_sha256": "x"}, log=lambda s: None)
    res = t.run()
    win = res["winner"]["overrides"]
    assert res["improved"] and res["winner"]["gain"] > 0
    assert win.get("a", 1.0) > 1.0 and win.get("b", 1.0) >= win.get("a", 1.0)   # compensated
    for p in PAIRS:
        assert res["winner"]["per_pair"][p]["folding_pct"] == 0.0


def test_cache_reused_across_commits_with_identical_registration_code(tmp_path):
    """Tuner-only edits (new commit, same registration code) keep cached evaluations."""
    calls = []
    fp = {"commit": "a" * 40, "diff_sha256": "x", "registration_code": "same"}
    Tuner(_spec(), pairs=PAIRS, evaluator=synthetic_evaluator(calls), out_dir=str(tmp_path / "c"),
          check_canonical=False, code_fingerprint=fp, log=lambda s: None).run()
    n = len(calls)
    calls2 = []
    fp2 = dict(fp, commit="b" * 40)
    Tuner(_spec(), pairs=PAIRS, evaluator=synthetic_evaluator(calls2), out_dir=str(tmp_path / "c"),
          check_canonical=False, code_fingerprint=fp2, log=lambda s: None).run()
    assert n > 0 and calls2 == []
    calls3 = []
    fp3 = dict(fp, commit="c" * 40, registration_code="changed")
    Tuner(_spec(), pairs=PAIRS, evaluator=synthetic_evaluator(calls3), out_dir=str(tmp_path / "c"),
          check_canonical=False, code_fingerprint=fp3, log=lambda s: None).run()
    assert len(calls3) > 0   # registration code changed: re-evaluated


def test_registration_code_fingerprint_ignores_tuner_files_and_version(tmp_path):
    import subprocess
    from syntx.benchmark.tune import registration_code_fingerprint
    repo = tmp_path / "r"
    pkg = repo / "src" / "syntx"
    (pkg / "benchmark").mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "1.0"\nX = 1\n')
    (pkg / "syn.py").write_text("A = 1\n")
    (pkg / "benchmark" / "tune.py").write_text("T = 1\n")
    g = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                                  check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    g("add", "."); g("commit", "-q", "-m", "1"); c1 = g("rev-parse", "HEAD")
    (pkg / "__init__.py").write_text('__version__ = "1.1"\nX = 1\n')
    (pkg / "benchmark" / "tune.py").write_text("T = 2\n")
    g("commit", "-qam", "2"); c2 = g("rev-parse", "HEAD")
    (pkg / "syn.py").write_text("A = 2\n")
    g("commit", "-qam", "3"); c3 = g("rev-parse", "HEAD")
    f1, f2, f3 = (registration_code_fingerprint(commit=c, pkg_dir=str(pkg)) for c in (c1, c2, c3))
    assert f1 == f2 != f3
    assert registration_code_fingerprint(pkg_dir=str(pkg)) == f3   # working tree == HEAD
    # through a symlinked path (macOS /var -> /private/var: tune --isolate worktrees)
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)
    assert registration_code_fingerprint(commit=c2, pkg_dir=str(link / "src" / "syntx")) == f2


def test_start_point_is_evaluated_first_and_centres_the_screen(tmp_path):
    calls = []
    res = _tuner(tmp_path, calls, start={"a": 2.0}).run()
    first_non_baseline = next(o for _, o in calls if o)
    assert first_non_baseline == {"a": 2.0}                   # evaluated right after the baseline
    screened = [o for _, o in calls if o.get("a") == 2.0 and "b" in o]
    assert screened                                           # other parameters screened around a=2.0
    assert res["defaults"]["a"] == 1.0                        # gains still relative to the defaults
    with pytest.raises(ValueError, match="not in the search space"):
        _tuner(tmp_path / "x", [], start={"zzz": 1}).run()
