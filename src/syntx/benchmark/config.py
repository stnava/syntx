"""
syntx.benchmark.config — per-method benchmark parameter blocks
==============================================================

``DEFAULT_BENCHMARK_CONFIG`` holds one block per method ('syn_config', 'gaussian_config',
'tvf_config', 'syngs_config', 'greedy_config') plus '_metadata'. The syn / tvf / syngs /
greedy blocks are declared to equal the corresponding function's defaults; that equality is
checked by the tests named in the comments, not by this module. Note that
``evaluate_mindboggle_pair`` runs syn / tvf / syngs / greedy with the function's own defaults
when it is given no ``config``; these blocks are then used only for the record's ``config`` /
``config_hash`` fields. The 'gaussian' arm, by contrast, takes its parameters (grad_step,
fluid_sigma, elastic_sigma, syn_fast_smooth, syn_metric, kernel_type, reg_iterations) from
'gaussian_config'.

Helpers: ``get_model_config`` (pick a model's block, filling missing keys from the defaults),
``syn_config_to_syn_kwargs`` (syn_config keys -> ``syntx.syn()`` keywords),
``compute_config_hash`` and ``validate_config_compatibility``.
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
        "alpha": 2.0,             # provisional (2-D study 2026-09-30); 3-D to be tuned
        "optimizer": "cfl",
        "grad_step": 1.0,
        "cfl_momentum": 0.9,
        "energy_weight": 0.001,
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
    """Return a deep copy of ``DEFAULT_BENCHMARK_CONFIG`` (safe to modify)."""
    return copy.deepcopy(DEFAULT_BENCHMARK_CONFIG)


def get_model_config(model: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the parameter block for ``model``, with missing keys filled from the defaults.

    Model names map to blocks as: 'syn' / 'sobolev' / 'syn_sobolev' -> 'syn_config';
    'gaussian' / 'syn_gaussian' / 'syn_mi' -> 'gaussian_config'; 'syngs' / 'geodesic' /
    'syn_gs' -> 'syngs_config'; 'tvf' -> 'tvf_config'; 'greedy' / 'syntx_greedy' /
    'greedy_regadam' / 'regadam_greedy' -> 'greedy_config'; anything else ->
    '<model>_config'.

    The block is taken from ``config`` (or ``DEFAULT_BENCHMARK_CONFIG`` when ``config`` is
    None or empty) as follows: the block itself if present; else ``config['params']`` plus
    the entry's top-level 'regularizer' / 'fast_smooth' if present (the
    ``syntx.benchmark.grid`` layout; stored as 'syn_regularizer' / 'syn_fast_smooth' for the
    SyN blocks); else a shallow copy of the whole
    ``config`` dict. If the block name exists in ``DEFAULT_BENCHMARK_CONFIG``, keys missing
    from the result are then filled from that default block, except keys whose alias is
    already set (e.g. no default 'fluid_sigma' next to a given 'flow_sigma').

    Parameters
    ----------
    model : str
        Model identifier (case-insensitive).
    config : dict, optional
        A full configuration (same layout as ``DEFAULT_BENCHMARK_CONFIG``, e.g. a loaded
        run_config.json) or a grid entry with 'params'.

    Returns
    -------
    dict
        A new dict (the input blocks are not modified; nested values are shared).
        For a model with no known block (e.g. 'ants', 'fireants', 'syn_regadam'): {} when
        ``config`` is None / empty, else (no matching key) a copy of the whole ``config``.
    """
    model_lower = str(model).lower()
    base_config = config if config else DEFAULT_BENCHMARK_CONFIG

    # Normalize model key
    if model_lower in ("syn", "sobolev", "syn_sobolev"):
        key = "syn_config"
    elif model_lower in ("gaussian", "syn_gaussian", "syn_mi"):
        key = "gaussian_config"
    elif model_lower in ("syngs", "geodesic", "syn_gs"):
        key = "syngs_config"
    elif model_lower == "tvf":
        key = "tvf_config"
    elif model_lower in ("greedy", "syntx_greedy", "greedy_regadam", "regadam_greedy"):
        key = "greedy_config"
    else:
        key = f"{model_lower}_config"

    # Extract user-provided or base config
    if key in base_config:
        model_cfg = dict(base_config[key])
    elif not config:
        model_cfg = {}               # no default block for this model: nothing configured
    elif "params" in base_config:
        model_cfg = dict(base_config["params"])
        # grid entries carry regularizer / fast_smooth next to 'params'; they are part of it
        # (under the block's own names: syn_* in the SyN blocks)
        prefix = "syn_" if key in ("syn_config", "gaussian_config") else ""
        for k in ("regularizer", "fast_smooth"):
            if k in base_config and prefix + k not in model_cfg:
                model_cfg[prefix + k] = base_config[k]
    else:
        model_cfg = dict(base_config)

    # Ensure fallbacks from DEFAULT_BENCHMARK_CONFIG if key exists there
    if key in DEFAULT_BENCHMARK_CONFIG:
        default_block = DEFAULT_BENCHMARK_CONFIG[key]
        # a default is skipped when an alias of it (same syntx keyword, e.g. fluid_sigma /
        # flow_sigma) is already set, so it cannot override the given value
        target = lambda k: _SYN_CONFIG_TO_SYN_KWARG.get(k, k)
        given = {target(k) for k in model_cfg}
        for k, v in default_block.items():
            if k not in model_cfg and target(k) not in given:
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

    Mapping (``_SYN_CONFIG_TO_SYN_KWARG``): fluid_sigma / flow_sigma -> flow_sigma,
    elastic_sigma / total_sigma -> total_sigma, lncc_radius -> syn_sampling,
    syn_regularizer -> regularizer, syn_fast_smooth -> fast_smooth,
    syn_use_analytical_gradients -> use_analytical_gradients, syn_inverse_method ->
    inverse_method, syn_formulation -> formulation; grad_step, in_loop_inv_steps, syn_metric,
    kernel_type, sobolev_alpha and reg_iterations keep their names. When two keys map to the
    same keyword (e.g. flow_sigma and fluid_sigma), the one later in the dict wins.

    Raises
    ------
    KeyError
        For a key with no mapping, so a config cannot silently carry a parameter that is not
        applied (``inverse_steps`` in the pre-5.4.46 config was such a key).
    """
    out = {}
    for k, v in syn_config.items():
        if k not in _SYN_CONFIG_TO_SYN_KWARG:
            raise KeyError(f"syn_config key {k!r} has no syntx.syn() mapping")
        out[_SYN_CONFIG_TO_SYN_KWARG[k]] = v
    return out


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """First 16 hex characters of the SHA-256 of ``config_dict`` serialized as JSON with
    sorted keys (non-JSON values via ``str``), so key order does not change the hash.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary to hash.

    Returns
    -------
    str
        16-character hexadecimal string.
    """
    # Normalize dictionary by serializing with sorted keys
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def validate_config_compatibility(cached_hash: Optional[str], current_hash: str) -> bool:
    """True if ``cached_hash`` is non-empty and equal to ``current_hash``."""
    if not cached_hash:
        return False
    return cached_hash == current_hash
