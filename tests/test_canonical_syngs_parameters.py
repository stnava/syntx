"""
The syngs benchmark path must run syntx.syngs with its own defaults (like greedy / syn):
config.py's syngs_config equals the signature defaults and
evaluate_mindboggle_pair(model="syngs") passes no tuning keyword of its own -- the
precondition for syntx.benchmark.tune. syngs reports an inverse error map, so interior
inverse statistics are available.
"""

import inspect
import math

import numpy as np
import pandas as pd
import ants

from syntx.benchmark.config import DEFAULT_BENCHMARK_CONFIG
from syntx.benchmark.evaluate import evaluate_mindboggle_pair
from syntx.benchmark.tune import METHODS, check_canonical_call
from syntx.syngs import syngs_registration

_CFG_TO_ARG = {"grad_step": "grad_step", "flow_sigma": "flow_sigma", "total_sigma": "total_sigma",
               "alpha": "alpha", "optimizer": "optimizer", "optimizer_lr": "optimizer_lr",
               "max_step_norm": "max_step_norm", "syn_metric": "syn_metric", "n_steps": "n_steps",
               "bootstrap_mode": "bootstrap_mode"}


def test_syngs_config_equals_signature_defaults():
    from syntx.syngs import default_alpha
    sig = inspect.signature(syngs_registration).parameters
    cfg = DEFAULT_BENCHMARK_CONFIG["syngs_config"]
    resolved = {name: p.default for name, p in sig.items()}
    resolved["alpha"] = default_alpha(3) if resolved["alpha"] is None else resolved["alpha"]
    bad = {k: (v, resolved[_CFG_TO_ARG[k]]) for k, v in cfg.items()
           if k in _CFG_TO_ARG and resolved[_CFG_TO_ARG[k]] != v}
    assert not bad, f"syngs_config disagrees with syntx.syngs defaults: {bad}"
    assert cfg["regularizer"] == METHODS["syngs"].defaults()["regularizer"]
    assert cfg["reg_iterations"] == [100, 100, 20]


def test_syngs_evaluator_path_is_canonical(tmp_path):
    data_dir = tmp_path / "data"
    rng = np.random.default_rng(0)
    for s in ("S1", "S2"):
        d = data_dir / "OASIS_volumes" / s
        d.mkdir(parents=True)
        ants.image_write(ants.from_numpy((rng.random((32, 32, 32)) * 100).astype(np.float32)),
                         str(d / "t1weighted_brain.nii.gz"))
        ants.image_write(ants.from_numpy(rng.integers(1, 4, (32, 32, 32)).astype(np.uint32)),
                         str(d / "labels.DKT31.manual.nii.gz"))
    csv = tmp_path / "pairs.csv"
    pd.DataFrame([{"cohort1": "OASIS", "subject1": "S1", "cohort2": "OASIS",
                   "subject2": "S2", "type": "intra"}]).to_csv(csv, index=False)
    overrides = {"reg_iterations": [2, 1, 1]}
    rec = evaluate_mindboggle_pair(0, "syngs", device="cpu", pairs_csv=str(csv),
                                   data_dir=str(data_dir), ants_baseline_dir=str(tmp_path),
                                   generate_report=False, verbose=False, **overrides)
    assert rec["status"] == "SUCCESS"
    check_canonical_call(METHODS["syngs"], rec, overrides)
    # p95 needs the error map (previously NaN for syngs); interior stats need a brain mask,
    # which this random-noise volume does not have.
    for k in ("syntx_inv_mean", "syntx_inv_p95", "syntx_inv_max"):
        assert math.isfinite(rec[k]), k
