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
        "version": "5.4.9",
        "description": "Authoritative syntx benchmark configuration (Sobolev SyN, Geodesic SyNGS, DSTI-1 TVF, Compositive Greedy)",
        "similarity_metric": "cc2",
        "reg_iterations": [100, 100, 20],
    },
    "syn_config": {
        "grad_step": 0.25,
        "fluid_sigma": 3.0,
        "elastic_sigma": 0.0,
        "lncc_radius": 2,
        "inverse_steps": 10,
        "syn_metric": "cc2",
        "syn_regularizer": "sobolev",
        "kernel_type": "sobolev",
        "sobolev_alpha": 1.5,
        "syn_fast_smooth": True,
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
    "tvf_config": {
        "optimizer": "reg_adam",
        "optimizer_lr": 1.2,
        "max_step_norm": 0.50,
        "tvf_flow_sigma": 1.0,
        "tvf_total_sigma": 0.035,
        "dsti_alpha": 0.035,
        "sobolev_alpha": 0.035,
        "tvf_cfl_momentum": 0.9,
        "tvf_n_time_steps": 3,
        "tvf_regularizer": "dsti1",
        "tvf_fast_smooth": False,
        "tvf_use_analytical_gradients": False,
        "tvf_antisymmetric": False,
        "multipoint_loss": [0.0, 0.5, 1.0],
        "similarity_metric": "cc2",
        "tvf_constant_speed": True,
        "tvf_constant_speed_relaxation": 0.1,
        "amp": False,
        "reg_iterations": [100, 100, 20],
    },
    "syngs_config": {
        "grad_step": 0.25,
        "flow_sigma": 3.0,
        "total_sigma": 0.0,
        "alpha": 0.35,
        "regularizer": "sobolev",
        "optimizer": "reg_adam",
        "optimizer_lr": 1.0,
        "max_step_norm": 0.20,
        "syn_metric": "cc2",
        "n_steps": 8,
        "reg_iterations": [100, 100, 20],
    },
    "greedy_config": {
        "grad_step": 0.50,
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
