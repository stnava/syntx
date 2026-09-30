"""
syntx.benchmark.config — Centralized Production Benchmark Configurations
========================================================================

Defines the authoritative single source of truth for benchmark hyperparameters,
model configurations, and configuration hash verification across all evaluation
pipelines and CLI tools.
"""

import copy
import hashlib
import json
from typing import Any, Dict, Optional


# Authoritative production defaults for syntx population benchmarks
DEFAULT_BENCHMARK_CONFIG: Dict[str, Any] = {
    "_metadata": {
        "version": "5.4.11",
        "description": "Authoritative syntx benchmark configuration (Sobolev SyN, Geodesic SyNGS, Gaussian TVF, Compositive Greedy)",
        "similarity_metric": "cc2",
        "reg_iterations": [100, 100, 20],
    },
    # Canonical: docs/provenance/best_parameters.json "syntx.syn/canonical_2026_09_29"
    # (automated tuning, pairs 77/44/0). Must equal syntx.syn()'s defaults -- enforced by
    # tests/test_canonical_parameters.py. (grad_step 0.35 / fluid_sigma 2.5 /
    # fast_smooth True were introduced without record in 9acc5fa, 2026-09-19.)
    "syn_config": {
        "grad_step": 0.4,
        "fluid_sigma": 2.4,
        "elastic_sigma": 0.0,
        "lncc_radius": 2,
        "in_loop_inv_steps": 10,
        "syn_metric": "cc2",
        "syn_regularizer": "sobolev",
        "kernel_type": "sobolev",
        "sobolev_alpha": 2.25,
        "syn_fast_smooth": False,
        "syn_use_analytical_gradients": False,
        "syn_inverse_method": "anderson",
        "syn_formulation": "eulerian",
        "reg_iterations": [100, 100, 20],
    },
    "gaussian_config": {
        "grad_step": 0.25,
        "fluid_sigma": 3.0,
        "elastic_sigma": 0.0,
        "lncc_radius": 2,
        "inverse_steps": 10,
        "syn_metric": "cc2",
        "syn_regularizer": "gaussian",
        "kernel_type": "gaussian",
        "syn_fast_smooth": False,
        "syn_use_analytical_gradients": False,
        "syn_inverse_method": "anderson",
        "syn_formulation": "eulerian",
        "reg_iterations": [100, 100, 20],
    },
    # Must equal syntx.tvf()'s defaults (tests/test_tvf_interface.py).
    "tvf_config": {
        "regularizer": "sobolev",
        "alpha": 2.0,             # provisional (2-D check 2026-09-30); to be set by the TVF tune
        "optimizer": "reg_adam",
        "optimizer_lr": 1.2,
        "max_step_norm": 0.5,
        "n_time_steps": 3,
        "multipoint_loss": [0.0, 0.5, 1.0],
        "fast_smooth": False,
        "syn_metric": "cc2",
        "syn_sampling": 2,
        "reg_iterations": [100, 100, 20],
    },
    # Must equal syntx.syngs()'s defaults (tests/test_canonical_parameters.py).
    "syngs_config": {
        "grad_step": 0.25,
        "total_sigma": 0.0,
        "alpha": 0.675,          # tuned 2026-09-30 (docs/provenance/tuning/syngs_2026-09-30.md)
        "regularizer": "sobolev",
        "optimizer": "reg_adam",
        "optimizer_lr": 1.0,
        "max_step_norm": 0.3,    # tuned 2026-09-30: +0.0159 mean Dice, 0 % folding on 77/44/0
        "syn_metric": "cc2",
        "n_steps": 8,
        "bootstrap_mode": "antithetic",
        "reg_iterations": [100, 100, 20],
    },
    # Must equal syntx.greedy()'s defaults (tests/test_canonical_parameters.py).
    "greedy_config": {
        "grad_step": 0.375,
        "flow_sigma": 1.8,
        "total_sigma": 0.28,
        "optimizer": "adam",
        "regadam_sigma": 0.8,
        "similarity_metric": "cc2",
        "reg_iterations": [100, 100, 20],
    },
}


def get_default_config() -> Dict[str, Any]:
    """Returns a deep copy of the standard benchmark configuration."""
    return copy.deepcopy(DEFAULT_BENCHMARK_CONFIG)


def get_model_config(model: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolves and extracts a model-specific configuration block.

    Parameters
    ----------
    model : str
        Model identifier (e.g. 'syn', 'syngs', 'tvf', 'greedy').
    config : dict, optional
        User configuration dictionary or loaded run_config.json. If None,
        falls back to ``DEFAULT_BENCHMARK_CONFIG``.

    Returns
    -------
    dict
        Model-specific parameter dictionary with defaults populated.
    """
    model_lower = str(model).lower()
    base_config = config if config else DEFAULT_BENCHMARK_CONFIG

    # Normalize model key
    if model_lower in ("syn", "sobolev", "syn_sobolev"):
        key = "syn_config"
    elif model_lower in ("gaussian", "syn_gaussian"):
        key = "gaussian_config"
    elif model_lower in ("syngs", "geodesic", "syn_gs"):
        key = "syngs_config"
    elif model_lower == "tvf":
        key = "tvf_config"
    elif model_lower in ("greedy", "syntx_greedy", "greedy_regadam"):
        key = "greedy_config"
    else:
        key = f"{model_lower}_config"

    # Extract user-provided or base config
    if key in base_config:
        model_cfg = dict(base_config[key])
    elif "params" in base_config:
        model_cfg = dict(base_config["params"])
    else:
        model_cfg = dict(base_config)

    # Ensure fallbacks from DEFAULT_BENCHMARK_CONFIG if key exists there
    if key in DEFAULT_BENCHMARK_CONFIG:
        default_block = DEFAULT_BENCHMARK_CONFIG[key]
        for k, v in default_block.items():
            if k not in model_cfg:
                model_cfg[k] = v

    return model_cfg


# syn_config key -> syntx.syn() keyword
_SYN_CONFIG_TO_SYN_KWARG = {
    "grad_step": "grad_step",
    "fluid_sigma": "flow_sigma",
    "flow_sigma": "flow_sigma",
    "elastic_sigma": "total_sigma",
    "total_sigma": "total_sigma",
    "lncc_radius": "syn_sampling",
    "in_loop_inv_steps": "in_loop_inv_steps",
    "syn_metric": "syn_metric",
    "syn_regularizer": "regularizer",
    "kernel_type": "kernel_type",
    "sobolev_alpha": "sobolev_alpha",
    "syn_fast_smooth": "fast_smooth",
    "syn_use_analytical_gradients": "use_analytical_gradients",
    "syn_inverse_method": "inverse_method",
    "syn_formulation": "formulation",
    "reg_iterations": "reg_iterations",
}


def syn_config_to_syn_kwargs(syn_config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a ``syn_config`` block into ``syntx.syn()`` keyword arguments.

    Unknown keys raise, so a config can never silently carry a parameter that is not
    applied (``inverse_steps`` in the pre-5.4.46 config was such a key).
    """
    out = {}
    for k, v in syn_config.items():
        if k not in _SYN_CONFIG_TO_SYN_KWARG:
            raise KeyError(f"syn_config key {k!r} has no syntx.syn() mapping")
        out[_SYN_CONFIG_TO_SYN_KWARG[k]] = v
    return out


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 fingerprint for a configuration block.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary to hash.

    Returns
    -------
    str
        16-character hexadecimal hash string.
    """
    # Normalize dictionary by serializing with sorted keys
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def validate_config_compatibility(cached_hash: Optional[str], current_hash: str) -> bool:
    """Validates whether a cached result matches the active configuration hash."""
    if not cached_hash:
        return False
    return cached_hash == current_hash
