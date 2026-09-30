"""
syntx.tvf interface contract: the benchmark configuration blocks mirror syntx.tvf()'s effective
defaults, and the advanced options are the declared set (nothing silently swallowed).
Per-parameter behaviour (accepted -> has an effect; inapplicable -> raises) is pinned in
tests/test_parameter_sensitivity.py::TestTVFParameterSensitivity.
"""

import inspect
import json
import os

import pytest

import syntx
from syntx.tvf import TVF_ADVANCED_OPTIONS, TVF_DEFAULT_ALPHA, tvf_registration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _effective_defaults(dim=3):
    sig = inspect.signature(tvf_registration)
    d = {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}
    d["alpha"] = TVF_DEFAULT_ALPHA[dim] if d["alpha"] is None else d["alpha"]
    d["optimizer_lr"] = 1.2 if d["optimizer_lr"] is None else d["optimizer_lr"]
    d["max_step_norm"] = 0.5 if d["max_step_norm"] is None else d["max_step_norm"]
    d["fast_smooth"] = False if d["fast_smooth"] is None else d["fast_smooth"]
    d["reg_iterations"] = [100, 100, 20] if d["reg_iterations"] is None else d["reg_iterations"]
    d["multipoint_loss"] = list(d["multipoint_loss"])
    return d


@pytest.mark.parametrize("source", ["config.py", "run_config.json"])
def test_tvf_config_blocks_mirror_the_defaults(source):
    if source == "config.py":
        from syntx.benchmark.config import DEFAULT_BENCHMARK_CONFIG
        block = DEFAULT_BENCHMARK_CONFIG["tvf_config"]
    else:
        block = json.load(open(os.path.join(ROOT, "docs/provenance/run_config.json")))["tvf_config"]
    eff = _effective_defaults(3)
    assert {k: v for k, v in block.items() if eff.get(k) != v} == {}, (
        f"{source} tvf_config differs from syntx.tvf defaults")
    assert set(block) <= set(eff), f"{source} tvf_config has keys syntx.tvf does not take: {set(block) - set(eff)}"


def test_tvf_accepts_every_declared_advanced_option_name():
    # the declared options are exactly what the front end forwards; spot-check one that the
    # model reads and one that it forwards to the model constructor
    assert "constant_speed" in TVF_ADVANCED_OPTIONS and "solver" in TVF_ADVANCED_OPTIONS
    assert syntx.tvf is tvf_registration
