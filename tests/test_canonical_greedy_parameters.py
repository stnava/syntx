"""
The greedy benchmark path must run syntx.greedy with its own defaults (like syntx.syn, see
test_canonical_syn_parameters.py): the config block equals the signature defaults and
evaluate_mindboggle_pair(model="greedy") passes no tuning keyword of its own -- the
precondition for syntx.benchmark.tune. greedy produces no inverse: its inverse metrics
are NaN.
"""

import inspect
import math

import numpy as np
import pandas as pd
import ants

from syntx.benchmark.config import DEFAULT_BENCHMARK_CONFIG
from syntx.benchmark.evaluate import evaluate_mindboggle_pair
from syntx.benchmark.tune import METHODS, check_canonical_call
from syntx.greedy import greedy_registration

_CFG_TO_ARG = {"grad_step": "learning_rate", "flow_sigma": "flow_sigma", "total_sigma": "total_sigma",
               "optimizer": "optimizer", "regadam_sigma": "regadam_sigma",
               "similarity_metric": "similarity_metric"}


def test_greedy_config_equals_signature_defaults():
    sig = inspect.signature(greedy_registration).parameters
    cfg = DEFAULT_BENCHMARK_CONFIG["greedy_config"]
    bad = {k: (v, sig[_CFG_TO_ARG[k]].default) for k, v in cfg.items()
           if k in _CFG_TO_ARG and sig[_CFG_TO_ARG[k]].default != v}
    assert not bad, f"greedy_config disagrees with syntx.greedy defaults: {bad}"
    assert cfg["reg_iterations"] == METHODS["greedy"].defaults()["reg_iterations"]


def test_greedy_evaluator_path_is_canonical(tmp_path):
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
    rec = evaluate_mindboggle_pair(0, "greedy", device="cpu", pairs_csv=str(csv),
                                   data_dir=str(data_dir), ants_baseline_dir=str(tmp_path),
                                   generate_report=False, verbose=False, **overrides)
    assert rec["status"] == "SUCCESS"
    check_canonical_call(METHODS["greedy"], rec, overrides)   # no hand-set parameters
    assert math.isnan(rec["syntx_inv_mean"]) and math.isnan(rec["syntx_inv_interior_max"])
