#!/usr/bin/env python
"""
syntx Benchmark: Single-Pair Registration Evaluation
=====================================================

Evaluates a single Mindboggle registration pair with full metrics collection.
Designed to be invoked both as a standalone CLI and as a subprocess worker
for the 90-pair orchestrator.

Usage
-----
    python scripts/run_single_pair.py \\
        --pair-idx 0 \\
        --model syn \\
        --device mps \\
        --config docs/provenance/run_config.json \\
        --out-json results/pair_000.json

    # Or with environment variable for data path:
    SYNTX_DATA_DIR=/path/to/mindboggle/volumes python scripts/run_single_pair.py ...

Output
------
Writes a JSON file with the following structure:

    {
        "pair_idx": 0,
        "model": "syn",
        "device": "mps",
        "fixed_id": "OASIS-TRT-20-17",
        "moving_id": "OASIS-TRT-20-16",
        "status": "SUCCESS",
        "dice_fixed": 0.6123,
        "dice_moving": 0.6197,
        "dice_sym": 0.6160,
        "folding_pct": 0.021,
        "min_jacobian": 0.412,
        "harmonic_energy": 0.2188,
        "bending_energy": 0.0902,
        "mattes_mi": -0.341,
        "lncc": -0.067,
        "runtime_seconds": 122.5,
        "config": { ... },
        "provenance": { ... }
    }
"""

import sys
import os
import argparse
import json
import time
import gc
import tempfile
import traceback
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("syntx.benchmark.single_pair")


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------
DEFAULT_PAIRS_CSV = "examples/pairs.csv"
DEFAULT_CONFIG = "docs/provenance/run_config.json"
DEFAULT_DATA_DIR_ENV = "SYNTX_DATA_DIR"
DEFAULT_DATA_DIR = "/Users/stnava/data/mindboggle/volumes"


def resolve_data_dir() -> str:
    """Resolves the Mindboggle data directory from environment or default.

    Returns
    -------
    str
        Absolute path to the data directory.

    Raises
    ------
    FileNotFoundError
        If the resolved directory does not exist.
    """
    data_dir = os.environ.get(DEFAULT_DATA_DIR_ENV, DEFAULT_DATA_DIR)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Set {DEFAULT_DATA_DIR_ENV} environment variable or ensure "
            f"the default path exists."
        )
    return data_dir


def load_pair(pairs_csv: str, pair_idx: int, data_dir: str) -> dict:
    """Loads a single image pair from the pairs CSV file.

    Parameters
    ----------
    pairs_csv : str
        Path to the pairs CSV file (columns: type, cohort1, subject1, cohort2, subject2).
    pair_idx : int
        Zero-based index of the pair to load.
    data_dir : str
        Base directory containing ``{cohort}_volumes/{subject}/`` subdirectories.

    Returns
    -------
    dict
        Dictionary with keys: 'fixed', 'moving', 'fixed_label', 'moving_label',
        'fixed_id', 'moving_id', 'pair_type'.

    Raises
    ------
    IndexError
        If pair_idx is out of range.
    FileNotFoundError
        If any required image file is missing.
    """
    import pandas as pd
    import ants

    df = pd.read_csv(pairs_csv)
    if pair_idx < 0 or pair_idx >= len(df):
        raise IndexError(
            f"pair_idx={pair_idx} out of range [0, {len(df) - 1}]. "
            f"CSV has {len(df)} pairs."
        )

    row = df.iloc[pair_idx]
    c1, s1 = row["cohort1"], row["subject1"]
    c2, s2 = row["cohort2"], row["subject2"]

    paths = {
        "fixed": os.path.join(data_dir, f"{c1}_volumes", s1, "t1weighted_brain.nii.gz"),
        "fixed_label": os.path.join(data_dir, f"{c1}_volumes", s1, "labels.DKT31.manual.nii.gz"),
        "moving": os.path.join(data_dir, f"{c2}_volumes", s2, "t1weighted_brain.nii.gz"),
        "moving_label": os.path.join(data_dir, f"{c2}_volumes", s2, "labels.DKT31.manual.nii.gz"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

    return {
        "fixed": ants.image_read(paths["fixed"]),
        "moving": ants.image_read(paths["moving"]),
        "fixed_label": ants.image_read(paths["fixed_label"]),
        "moving_label": ants.image_read(paths["moving_label"]),
        "fixed_id": s1,
        "moving_id": s2,
        "pair_type": str(row.get("type", "unknown")),
    }


def load_config(config_path: str) -> dict:
    """Loads and validates a benchmark configuration JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    json.JSONDecodeError
        If the config file is not valid JSON.
    ValueError
        If required keys are missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if "syn_config" not in config and "tvf_config" not in config:
        raise ValueError(
            "Config must contain at least 'syn_config' or 'tvf_config'. "
            f"Found keys: {list(config.keys())}"
        )

    return config


def run_registration(data: dict, model: str, device: str, config: dict) -> dict:
    """Runs a single registration and computes all metrics.

    Parameters
    ----------
    data : dict
        Image pair dictionary from ``load_pair()``.
    model : str
        Model type: 'syn' or 'tvf'.
    device : str
        Compute device: 'cpu', 'mps', or 'cuda'.
    config : dict
        Full configuration dictionary from ``load_config()``.

    Returns
    -------
    dict
        Complete metrics dictionary with all standardized keys.
    """
    import torch
    import numpy as np
    import ants
    import syntx
    from syntx.deformation_metrics import compute_bidirectional_dice

    fixed = data["fixed"]
    moving = data["moving"]
    fixed_label = data["fixed_label"]
    moving_label = data["moving_label"]

    if model == "syn":
        cfg = config.get("syn_config", {})
    elif model == "tvf":
        cfg = config.get("tvf_config", {})
    else:
        cfg = {}

    # ---- 1. Deterministic Affine Initialization (always on CPU) ----
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info("Computing deterministic robust affine (CPU)...")
    t_aff_start = time.time()
    res_aff = syntx.robust_affine(
        fixed=fixed, moving=moving,
        mode="pytorch", device="cpu",
        multi_start=True, verbose=False,
        n_starts=cfg.get("n_starts", 4),
        cone_angles_deg=cfg.get("cone_angles_deg", [-25.0, -15.0, -5.0, 0.0, 5.0, 15.0, 25.0])
    )
    initial_transform = res_aff["fwdtransforms"][0]
    t_aff = time.time() - t_aff_start
    logger.info(f"Affine alignment completed in {t_aff:.1f}s")
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- 2. Nonlinear Registration ----
    t_reg_start = time.time()

    if model == "syn":
        logger.info(f"Running SyN on {device} with regularizer={cfg.get('syn_regularizer', 'gaussian')}...")
        reg = syntx.syn(
            fixed=fixed,
            moving=moving,
            initial_transform=initial_transform,
            backend="pytorch",
            device=device,
            grad_step=cfg.get("grad_step", 0.25),
            flow_sigma=cfg.get("fluid_sigma", 3.0),
            total_sigma=cfg.get("elastic_sigma", 0.0),
            similarity_metric=cfg.get("syn_metric", "lncc"),
            syn_sampling=cfg.get("lncc_radius", 2),
            inverse_steps=cfg.get("inverse_steps", 10),
            regularizer=cfg.get("syn_regularizer", "gaussian"),
            fast_smooth=cfg.get("syn_fast_smooth", False),
            use_analytical_gradients=cfg.get("syn_use_analytical_gradients", False),
            inverse_method=cfg.get("syn_inverse_method", "anderson"),
            formulation=cfg.get("syn_formulation", "eulerian"),
            reg_iterations=cfg.get("reg_iterations", [100, 100, 20]),
            antisymmetric=True,
            verbose=False,
        )
    elif model == "tvf":
        cfg = config.get("tvf_config", {})
        logger.info(f"Running TVF on {device} with regularizer={cfg.get('tvf_regularizer', 'gaussian')}...")
        reg = syntx.tvf(
            fixed=fixed,
            moving=moving,
            initial_transform=initial_transform,
            backend="pytorch",
            device=device,
            grad_step=cfg.get("tvf_grad_step", 0.211),
            flow_sigma=cfg.get("tvf_flow_sigma", 0.0),
            total_sigma=cfg.get("tvf_total_sigma", 0.2),
            regularizer=cfg.get("tvf_regularizer", "gaussian"),
            fast_smooth=cfg.get("tvf_fast_smooth", False),
            use_analytical_gradients=cfg.get("tvf_use_analytical_gradients", False),
            antisymmetric=cfg.get("tvf_antisymmetric", True),
            cfl_momentum=cfg.get("tvf_cfl_momentum", 0.9),
            n_time_steps=cfg.get("tvf_n_time_steps", 3),
            constant_speed=cfg.get("tvf_constant_speed", True),
            constant_speed_relaxation=cfg.get("tvf_constant_speed_relaxation", 0.10),
            multipoint_loss=[0.0, 0.5, 1.0],
            reg_iterations=cfg.get("reg_iterations", [80, 80, 20]),
            verbose=False,
        )
    elif model == "ants_syn":
        logger.info("Running ANTs C++ SyN on CPU (ITK)...")
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            initial_transform=initial_transform,
            type_of_transform="SyN",
            syn_metric="CC",
            grad_step=0.25,
            flow_sigma=3.0,
            total_sigma=0.0,
            syn_sampling=2,
            reg_iterations=(100, 100, 20),
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'syn', 'tvf', or 'ants_syn'.")

    t_reg = time.time() - t_reg_start
    t_total = t_aff + t_reg
    logger.info(f"Registration completed in {t_reg:.1f}s (total with affine: {t_total:.1f}s)")

    fwdtransforms = reg["fwdtransforms"]
    invtransforms = reg["invtransforms"]
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- 3. Compute All Metrics ----
    metrics = {}

    # 3a. Bidirectional Dice
    try:
        dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
            fixed_label, moving_label, fixed, moving,
            fwdtransforms, invtransforms,
        )
        metrics["dice_fixed"] = float(dice_fixed)
        metrics["dice_moving"] = float(dice_moving)
        metrics["dice_sym"] = float(dice_sym)
    except Exception as e:
        logger.warning(f"Failed to compute Dice: {e}")
        metrics["dice_fixed"] = float("nan")
        metrics["dice_moving"] = float("nan")
        metrics["dice_sym"] = float("nan")

    # 3b. Topological and Energy Metrics
    try:
        warp_path = next(
            (p for p in fwdtransforms if isinstance(p, str) and p.endswith(".nii.gz")),
            None,
        )
        if warp_path is not None:
            warp_img = ants.image_read(warp_path)
            dim = warp_img.dimension
            spc = warp_img.spacing

            # Jacobian
            jac_img = ants.create_jacobian_determinant_image(fixed, warp_img, do_log=False)
            jac_np = jac_img.numpy()
            mask = ants.get_mask(fixed).numpy() > 0

            metrics["folding_pct"] = float(np.mean(jac_np[mask] <= 0) * 100.0) if mask.any() else 0.0
            metrics["min_jacobian"] = float(jac_np[mask].min()) if mask.any() else 1.0

            # Harmonic and Bending Energy
            warp_np = warp_img.numpy()
            gradient_list = [
                np.gradient(warp_np[..., k], *spc, axis=range(dim))
                for k in range(dim)
            ]

            total_hrm = 0.0
            total_bnd = 0.0
            for k in range(dim):
                for j in range(dim):
                    grad_kj = gradient_list[k][j]
                    total_hrm += float(np.mean(grad_kj ** 2))

                    grad2_kj = np.gradient(grad_kj, *spc, axis=range(dim))
                    for i in range(dim):
                        total_bnd += float(np.mean(grad2_kj[i] ** 2))

            metrics["harmonic_energy"] = total_hrm
            metrics["bending_energy"] = total_bnd
        else:
            metrics["folding_pct"] = 0.0
            metrics["min_jacobian"] = 1.0
            metrics["harmonic_energy"] = 0.0
            metrics["bending_energy"] = 0.0
    except Exception as e:
        logger.warning(f"Failed to compute topological metrics: {e}")
        metrics["folding_pct"] = float("nan")
        metrics["min_jacobian"] = float("nan")
        metrics["harmonic_energy"] = float("nan")
        metrics["bending_energy"] = float("nan")

    # 3c. Image Similarity Metrics
    try:
        mi_warped = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=fwdtransforms
        )
        metrics["mattes_mi"] = float(syntx.image_compare(fixed, mi_warped, "mattes_mi"))
        metrics["lncc"] = float(syntx.image_compare(fixed, mi_warped, "lncc"))
    except Exception as e:
        logger.warning(f"Failed to compute image similarity metrics: {e}")
        metrics["mattes_mi"] = float("nan")
        metrics["lncc"] = float("nan")

    metrics["runtime_seconds"] = round(t_total, 2)
    metrics["runtime_affine_seconds"] = round(t_aff, 2)
    metrics["runtime_nonlinear_seconds"] = round(t_reg, 2)

    # ---- 4. Memory Cleanup ----
    del reg
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return metrics


def write_result_atomic(result: dict, out_path: str) -> None:
    """Atomically writes a result dictionary to a JSON file.

    Uses write-to-temp + os.replace for crash safety.

    Parameters
    ----------
    result : dict
        Result dictionary to serialize.
    out_path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(out_path)),
        prefix=".result_",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    """CLI entry point for single-pair benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Run a single Mindboggle pair registration benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pair-idx 0 --model syn --device mps --config docs/provenance/run_config.json
  %(prog)s --pair-idx 5 --model tvf --device cpu --out-json results/pair_005_tvf.json
  SYNTX_DATA_DIR=/data/mindboggle %(prog)s --pair-idx 0 --model syn --device mps
        """,
    )
    parser.add_argument(
        "--pair-idx", type=int, required=True,
        help="Zero-based pair index from pairs.csv (0-89).",
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["syn", "tvf", "ants_syn"],
        help="Registration model to evaluate.",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Compute device (default: mps).",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help=f"Path to JSON config file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--out-json", type=str, default=None,
        help="Output JSON path (default: results/pair_{idx:03d}_{model}.json).",
    )
    parser.add_argument(
        "--pairs-csv", type=str, default=DEFAULT_PAIRS_CSV,
        help=f"Path to pairs CSV (default: {DEFAULT_PAIRS_CSV}).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=f"Mindboggle data directory (overrides {DEFAULT_DATA_DIR_ENV} env var).",
    )

    args = parser.parse_args()

    # Resolve paths
    data_dir = args.data_dir or resolve_data_dir()

    if args.out_json is None:
        os.makedirs("results", exist_ok=True)
        args.out_json = f"results/pair_{args.pair_idx:03d}_{args.model}.json"

    # Build result record
    result = {
        "pair_idx": args.pair_idx,
        "model": args.model,
        "device": args.device,
        "status": "RUNNING",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    try:
        # Load config
        config = load_config(args.config)
        result["config"] = config.get(f"{args.model}_config", {})

        # Load images
        logger.info(f"Loading pair {args.pair_idx}...")
        data = load_pair(args.pairs_csv, args.pair_idx, data_dir)
        result["fixed_id"] = data["fixed_id"]
        result["moving_id"] = data["moving_id"]
        result["pair_type"] = data["pair_type"]
        logger.info(
            f"Pair {args.pair_idx}: {data['fixed_id']} -> {data['moving_id']} "
            f"({data['pair_type']})"
        )

        # Run registration + metrics
        metrics = run_registration(data, args.model, args.device, config)
        result.update(metrics)
        result["status"] = "SUCCESS"

        logger.info(
            f"RESULT | Pair {args.pair_idx} | "
            f"Dice={result['dice_sym']:.4f} | "
            f"Fold={result['folding_pct']:.3f}% | "
            f"Time={result['runtime_seconds']:.1f}s"
        )

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"FAILED pair {args.pair_idx}: {e}")

    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Write result atomically
    write_result_atomic(result, args.out_json)
    logger.info(f"Result written to {args.out_json}")

    # Machine-parseable stdout line for orchestrators
    if result["status"] == "SUCCESS":
        print(
            f"DONE:{args.pair_idx}:{args.model}:{args.device}:"
            f"{result['dice_sym']:.4f}:{result['folding_pct']:.4f}"
        )
    else:
        print(f"FAIL:{args.pair_idx}:{args.model}:{args.device}:{result.get('error', 'unknown')}")

    sys.exit(0 if result["status"] == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
