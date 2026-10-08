"""
Regression tests for continuum-mechanics regularizers in syntx benchmark infrastructure:
configuration mapping, model set expansion, search space validation, and 2D/3D evaluator dispatch.
"""

import pytest
import ants
import numpy as np
import torch

from syntx.benchmark.config import (
    DEFAULT_BENCHMARK_CONFIG,
    get_model_config,
    syn_config_to_syn_kwargs,
)
from syntx.benchmark.orchestrator import expand_model_set
from syntx.benchmark.tune import METHODS, Param, twod_evaluator


def test_continuum_config_blocks_exist():
    assert "syn_navier_config" in DEFAULT_BENCHMARK_CONFIG
    assert "syn_solenoidal_config" in DEFAULT_BENCHMARK_CONFIG
    assert "syn_divcurl_config" in DEFAULT_BENCHMARK_CONFIG
    assert "tvf_navier_config" in DEFAULT_BENCHMARK_CONFIG


def test_continuum_config_mapping():
    navier_cfg = get_model_config("syn_navier")
    assert navier_cfg["syn_regularizer"] == "navier"
    assert navier_cfg["poisson_ratio"] == 0.35
    assert navier_cfg["s"] == 2.0
    assert navier_cfg["sobolev_alpha"] == 1.5

    kwargs = syn_config_to_syn_kwargs(navier_cfg)
    assert kwargs["regularizer"] == "navier"
    assert kwargs["poisson_ratio"] == 0.35
    assert kwargs["s"] == 2.0
    assert kwargs["sobolev_alpha"] == 1.5

    divcurl_cfg = get_model_config("syn_divcurl")
    kwargs_dc = syn_config_to_syn_kwargs(divcurl_cfg)
    assert kwargs_dc["regularizer"] == "div_curl"
    assert kwargs_dc["beta"] == 4.0
    assert kwargs_dc["gamma"] == 0.8
    assert kwargs_dc["h3_envelope"] is True


def test_continuum_model_sets_expansion():
    syn_cont = expand_model_set("continuum_syn")
    assert "syn_navier" in syn_cont
    assert "syn_divcurl" in syn_cont

    top5 = expand_model_set("continuum_top5")
    assert top5 == ["sobolev", "syn_divcurl", "syn_hyperelastic", "syn_navier", "gaussian"]

    all_cont = expand_model_set("all_continuum")
    assert "syn_navier" in all_cont
    assert "syn_hyperelastic" in all_cont
    assert "gaussian" in all_cont


def test_tune_search_space_continuum_params():
    syn_spec = METHODS["syn"]
    reg_param = next(p for p in syn_spec.space if p.name == "regularizer")
    assert "navier" in reg_param.values
    assert "solenoidal" in reg_param.values
    assert "div_curl" in reg_param.values

    # Test conditional activation
    nu_param = next(p for p in syn_spec.space if p.name == "poisson_ratio")
    assert nu_param.active({"regularizer": "navier"})
    assert not nu_param.active({"regularizer": "sobolev"})

    beta_param = next(p for p in syn_spec.space if p.name == "beta")
    assert beta_param.active({"regularizer": "div_curl"})
    assert not beta_param.active({"regularizer": "navier"})


def test_twod_evaluator_navier_smoke():
    """Verify that twod_evaluator runs cleanly on 2D with Navier regularizer."""
    spec = METHODS["syn"]
    evaluator = twod_evaluator(spec, device="cpu")
    overrides = {
        "regularizer": "navier",
        "poisson_ratio": 0.48,
        "reg_iterations": [2, 1],
    }
    metrics, rec = evaluator(0, overrides)
    assert "dice_sym" in metrics
    assert "folding_pct" in metrics
    assert metrics["dice_sym"] > 0.0
