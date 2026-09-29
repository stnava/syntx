"""
The canonical syntx.syn parameters must agree everywhere they are written down.

Canonical source: docs/provenance/best_parameters.json,
"90pair_population_benchmark_sobolev_mps" (fast_smooth False per project decision).
syntx.syn()'s defaults must reproduce it, and the benchmark configs
(src/syntx/benchmark/config.py, docs/provenance/run_config.json) must equal those
defaults. Motivation: config.py silently drifted to grad_step 0.35 / fluid_sigma 2.5 /
fast_smooth True (9acc5fa), and syntx.syn used in_loop_inv_steps 6 vs the record's 10.

syntx.syn resolves several defaults inside its body, so they are captured behaviourally:
SyNTo.fit is intercepted on a tiny pair and the run is aborted before any optimisation.
"""

import importlib
import json
import os

import numpy as np
import pytest
import ants

import syntx
from syntx.benchmark.config import DEFAULT_BENCHMARK_CONFIG, syn_config_to_syn_kwargs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORD = "90pair_population_benchmark_sobolev_mps"
syn_module = importlib.import_module("syntx.syn")  # the module (syntx.syn is the function)

# The Bessel ('bessel') and 'sobolev' kernel_type values select the same spatial filter.
_KERNEL_EQUIV = {"bessel": "bessel", "sobolev": "bessel"}


class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def syn_defaults():
    """Effective syntx.syn() defaults, as received by SyNTo.fit for a 3D pair."""
    captured = {}

    def _capture(self, I, J, **kw):
        captured.update(kw)
        captured["_model"] = self
        raise _Stop

    orig = syn_module.SyNTo.fit
    syn_module.SyNTo.fit = _capture
    try:
        arr = np.random.default_rng(0).random((20, 20, 20)).astype(np.float32)
        img = ants.from_numpy(arr)
        with pytest.raises(_Stop):
            syntx.syn(fixed=img, moving=img, initial_transform="identity", device="cpu",
                      verbose=False)
    finally:
        syn_module.SyNTo.fit = orig

    m = captured["_model"]
    alpha = captured.get("sobolev_alpha")
    return {
        "grad_step": captured["cfl_voxels"],
        # registration() treats flow_sigma as an ITK variance; the model stores sigma
        "flow_sigma": m.fluid_sigma ** 2,
        "total_sigma": m.elastic_sigma,
        "syn_sampling": captured["lncc_radius"],
        "in_loop_inv_steps": m.in_loop_inv_steps,
        "syn_metric": captured["similarity_metric"],
        "regularizer": captured["regularizer"],
        # fit() falls back to flow_sigma / 2 when no alpha is given
        "sobolev_alpha": float(alpha) if alpha is not None else m.fluid_sigma / 2.0,
        "fast_smooth": bool(captured["fast_smooth"]),
        "use_analytical_gradients": bool(captured["use_analytical_gradients"]),
        "inverse_method": m.inverse_method,
        "formulation": m.formulation,
        "reg_iterations": list(captured["epochs_per_level"]),
        "kernel_type": m.kernel_type,
    }


def _norm(k, v):
    if k == "kernel_type":
        return _KERNEL_EQUIV.get(v, v)
    if k == "reg_iterations":
        return list(v)
    if isinstance(v, float):
        return round(v, 9)
    return v


def _assert_matches(tag, expected, defaults):
    bad = {k: (v, defaults[k]) for k, v in expected.items()
           if _norm(k, v) != _norm(k, defaults[k])}
    assert not bad, f"{tag} disagrees with syntx.syn defaults (expected, default): {bad}"


def test_syn_defaults_match_canonical_record(syn_defaults):
    with open(os.path.join(ROOT, "docs/provenance/best_parameters.json")) as f:
        params = json.load(f)["syntx.syn"][RECORD]["parameters"]
    assert params.get("fast_smooth") is False  # recorded decision
    _assert_matches(RECORD, params, syn_defaults)


def test_benchmark_config_matches_syn_defaults(syn_defaults):
    _assert_matches("config.py syn_config",
                    syn_config_to_syn_kwargs(DEFAULT_BENCHMARK_CONFIG["syn_config"]),
                    syn_defaults)


def test_run_config_json_matches_syn_defaults(syn_defaults):
    with open(os.path.join(ROOT, "docs/provenance/run_config.json")) as f:
        syn_config = json.load(f)["syn_config"]
    _assert_matches("run_config.json syn_config", syn_config_to_syn_kwargs(syn_config),
                    syn_defaults)


def test_syn_config_rejects_unmapped_keys():
    with pytest.raises(KeyError):
        syn_config_to_syn_kwargs({"inverse_steps": 10})


def test_optimizer_lr_defaults_per_optimizer():
    """RegAdam / Adam family default to 0.5 (max step = 0.5 * grad_step); the old 1e-3
    default was a sentinel for grad_step -- a max step of grad_step**2 that left
    optimizer='reg_adam' badly under-registered (mbhard Dice 0.509 vs 0.608).
    rprop / sgd keep 1e-3; explicit values are used as given."""
    from syntx.syn import resolve_optimizer_lr
    assert resolve_optimizer_lr("reg_adam") == 0.5
    assert resolve_optimizer_lr("regadam") == 0.5
    assert resolve_optimizer_lr("adam") == 0.5
    assert resolve_optimizer_lr("cfl") == 1e-3
    assert resolve_optimizer_lr("rprop") == 1e-3
    assert resolve_optimizer_lr("sgd") == 1e-3
    assert resolve_optimizer_lr("reg_adam", 1.0) == 1.0
    assert resolve_optimizer_lr("reg_adam", 1e-3) == 1e-3   # no longer a sentinel
