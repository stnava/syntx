"""
Canonical defaults, one test for every method declared in syntx.benchmark.tune.METHODS.

For each method the *effective* defaults (probed at the model's fit(), so hidden
body-resolved defaults are included; else the signature default; else the declared hidden
default) must equal:
  - config.py DEFAULT_BENCHMARK_CONFIG[<config_block>]
  - docs/provenance/run_config.json[<run_config_block>]
  - the canonical record in docs/provenance/best_parameters.json named by
    docs/provenance/canonical.json (where one exists)
and the standard evaluator path must call the method with its own defaults (no hand-set
tuning keywords) -- the precondition for syntx.benchmark.tune. Adding a method = adding its
declaration in tune._CANONICAL; this test then covers it.
"""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest
import ants

from syntx.benchmark.config import DEFAULT_BENCHMARK_CONFIG
from syntx.benchmark.tune import (
    CANONICAL_POINTER,
    METHODS,
    canonical_mismatches,
    check_canonical_call,
    expected_defaults,
    probe_effective_defaults,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANONICAL_METHODS = sorted(n for n, s in METHODS.items() if s.config_block)


@pytest.fixture(scope="module")
def expected():
    cache = {}

    def get(name):
        if name not in cache:
            spec = METHODS[name]
            cache[name] = expected_defaults(spec, probe_effective_defaults(spec))
        return cache[name]
    return get


def test_every_tunable_method_is_canonical():
    for name, spec in METHODS.items():
        assert spec.config_block and spec.function_file and spec.resolved, name


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_config_py_block_matches_effective_defaults(method, expected):
    spec = METHODS[method]
    bad = canonical_mismatches(spec, DEFAULT_BENCHMARK_CONFIG[spec.config_block], expected(method))
    assert not bad, f"config.py {spec.config_block} vs {method} defaults (value, expected): {bad}"


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_run_config_json_block_matches_effective_defaults(method, expected):
    spec = METHODS[method]
    with open(os.path.join(ROOT, "docs/provenance/run_config.json")) as f:
        block = json.load(f).get(spec.run_config_block)
    if block is None:
        pytest.skip(f"run_config.json has no {spec.run_config_block}")
    bad = canonical_mismatches(spec, block, expected(method))
    assert not bad, f"run_config.json {spec.run_config_block} vs {method} defaults: {bad}"


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_canonical_record_matches_effective_defaults(method, expected):
    with open(os.path.join(ROOT, CANONICAL_POINTER)) as f:
        pointer = json.load(f)
    if method not in pointer:
        pytest.skip(f"no canonical record for {method} yet")
    with open(os.path.join(ROOT, "docs/provenance/best_parameters.json")) as f:
        data = json.load(f)
    group, key = pointer[method].split("/", 1)
    params = data[group][key]["parameters"]
    bad = canonical_mismatches(METHODS[method], params, expected(method), keys_are_config=False)
    assert not bad, f"{pointer[method]} parameters vs {method} defaults: {bad}"


@pytest.fixture(scope="module")
def mock_mindboggle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mb")
    data_dir = tmp / "data"
    rng = np.random.default_rng(0)
    for s in ("S1", "S2"):
        d = data_dir / "OASIS_volumes" / s
        d.mkdir(parents=True)
        ants.image_write(ants.from_numpy((rng.random((32, 32, 32)) * 100).astype(np.float32)),
                         str(d / "t1weighted_brain.nii.gz"))
        ants.image_write(ants.from_numpy(rng.integers(1, 4, (32, 32, 32)).astype(np.uint32)),
                         str(d / "labels.DKT31.manual.nii.gz"))
    csv = tmp / "pairs.csv"
    pd.DataFrame([{"cohort1": "OASIS", "subject1": "S1", "cohort2": "OASIS",
                   "subject2": "S2", "type": "intra"}]).to_csv(csv, index=False)
    return tmp, data_dir, csv


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_evaluator_path_calls_method_with_its_defaults(method, mock_mindboggle, monkeypatch):
    from syntx.benchmark.evaluate import evaluate_mindboggle_pair
    tmp, data_dir, csv = mock_mindboggle
    monkeypatch.chdir(tmp)          # the affine cache (results/canonical_affines) is cwd-relative
    spec = METHODS[method]
    overrides = {"reg_iterations": [2, 1, 1]}
    rec = evaluate_mindboggle_pair(0, spec.model, device="cpu", pairs_csv=str(csv),
                                   data_dir=str(data_dir), ants_baseline_dir=str(tmp),
                                   generate_report=False, verbose=False, **overrides)
    assert rec["status"] == "SUCCESS"
    check_canonical_call(spec, rec, overrides)
    if spec.has_inverse:  # needs the error map (p95), not a brain mask
        for k in ("syntx_inv_mean", "syntx_inv_p95", "syntx_inv_max"):
            assert math.isfinite(rec[k]), k
    else:
        assert math.isnan(rec["syntx_inv_mean"])


# ---- syn-specific (moved from test_canonical_syn_parameters.py) -------------------------
def test_syn_optimizer_lr_defaults_per_optimizer():
    """RegAdam / Adam family default 0.5 (max step = 0.5 * grad_step); rprop / sgd 1e-3."""
    from syntx.syn import resolve_optimizer_lr
    assert resolve_optimizer_lr("reg_adam") == 0.5 and resolve_optimizer_lr("adam") == 0.5
    assert resolve_optimizer_lr("cfl") == 1e-3 and resolve_optimizer_lr("sgd") == 1e-3
    assert resolve_optimizer_lr("reg_adam", 1.0) == 1.0
    assert resolve_optimizer_lr("reg_adam", 1e-3) == 1e-3   # no longer a sentinel


def test_syn_config_rejects_unmapped_keys():
    from syntx.benchmark.config import syn_config_to_syn_kwargs
    with pytest.raises(KeyError):
        syn_config_to_syn_kwargs({"inverse_steps": 10})
