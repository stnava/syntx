"""
syntx.benchmark.msd — Automated Medical Segmentation Decathlon Evaluation Suite
==============================================================================

Automates end-to-end evaluation of syntx registration methods across Decathlon tasks:
- Modality diagnosis verification
- Anatomical region diagnosis verification
- Automated zero-effort auto_reg registration
- Bidirectional segmentation label DICE scoring
- Grid topology & folding quantification
"""

import os
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import ants

import syntx
from syntx.data.msd import MSD_TASKS, get_msd_task_info
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics


def _prepare_image_and_label(img_path: str, lbl_path: str, channel: int = 0) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
    """Helper to read 3D/4D image and corresponding segmentation label map."""
    img = ants.image_read(img_path)
    lbl = ants.image_read(lbl_path)

    if img.dimension == 4:
        # Extract specific 3D channel
        arr = img.numpy()[..., channel]
        img = ants.from_numpy(arr, origin=img.origin[:3], spacing=img.spacing[:3], direction=img.direction[:3, :3])

    if lbl.dimension == 4:
        arr_l = lbl.numpy()[..., 0]
        lbl = ants.from_numpy(arr_l, origin=lbl.origin[:3], spacing=lbl.spacing[:3], direction=lbl.direction[:3, :3])

    return img, lbl


def evaluate_msd_pair(
    task_dir: str,
    fixed_idx: int,
    moving_idx: int,
    reg_iterations: Optional[List[int]] = None,
    affine_iterations: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluates automated registration on a pair of cases from an MSD task.

    Parameters:
    -----------
    task_dir : str
        Directory path containing unpacked MSD task (with dataset.json).
    fixed_idx : int
        Index of fixed target case in training manifest.
    moving_idx : int
        Index of moving source case in training manifest.
    reg_iterations : list of int, optional
        Deformable registration iterations (default: [40, 20, 10]).
    affine_iterations : list of int, optional
        Affine iterations (default: [20, 10]).
    verbose : bool, default=False
        Whether to print verbose optimization progress.

    Returns:
    --------
    dict
        Comprehensive evaluation results including DICE, Jacobian metrics, and diagnosis.
    """
    meta_path = os.path.join(task_dir, "dataset.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"dataset.json not found in {task_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    task_name = meta.get("name", os.path.basename(task_dir.rstrip("/")))
    training_cases = meta.get("training", [])
    if fixed_idx >= len(training_cases) or moving_idx >= len(training_cases):
        raise IndexError(f"Case indices ({fixed_idx}, {moving_idx}) exceed dataset size ({len(training_cases)})")

    fix_case = training_cases[fixed_idx]
    mov_case = training_cases[moving_idx]

    fix_img_p = os.path.normpath(os.path.join(task_dir, fix_case["image"]))
    fix_lbl_p = os.path.normpath(os.path.join(task_dir, fix_case["label"]))
    mov_img_p = os.path.normpath(os.path.join(task_dir, mov_case["image"]))
    mov_lbl_p = os.path.normpath(os.path.join(task_dir, mov_case["label"]))

    fi, fl = _prepare_image_and_label(fix_img_p, fix_lbl_p)
    mi, ml = _prepare_image_and_label(mov_img_p, mov_lbl_p)

    # Initial overlap (unaligned baseline)
    d_fix_0, d_mov_0, d_sym_0 = compute_bidirectional_dice(fl, ml, fi, mi, [], [], [])

    # Run auto_reg with autonomous diagnosis & policy synthesis
    t0 = time.time()
    res = syntx.auto_reg(
        fixed=fi,
        moving=mi,
        fixed_label=fl,
        moving_label=ml,
        reg_iterations=reg_iterations or [40, 20, 10],
        affine_iterations=affine_iterations or [20, 10],
        verbose=verbose
    )
    t_elapsed = time.time() - t0

    metrics = res.get("metrics", {})
    dice_sym = metrics.get("dice_sym", float("nan"))
    dice_fix = metrics.get("dice_fixed", float("nan"))
    dice_mov = metrics.get("dice_moving", float("nan"))
    folding_pct = metrics.get("folding_pct", 0.0)
    min_jac = metrics.get("jac_min", 1.0)
    diag = metrics.get("diagnosis", {})

    return {
        "task_name": task_name,
        "fixed_case": os.path.basename(fix_img_p),
        "moving_case": os.path.basename(mov_img_p),
        "initial_dice_sym": float(d_sym_0),
        "final_dice_sym": float(dice_sym),
        "final_dice_fixed": float(dice_fix),
        "final_dice_moving": float(dice_mov),
        "dice_gain": float(dice_sym - d_sym_0),
        "folding_pct": float(folding_pct),
        "min_jac": float(min_jac),
        "time_seconds": float(t_elapsed),
        "diagnosed_relationship": diag.get("relationship", "UNKNOWN"),
        "diagnosed_fixed_anatomy": diag.get("fixed_anatomy", "UNKNOWN"),
        "diagnosed_fixed_modality": diag.get("fixed_modality", "UNKNOWN"),
        "policy_explanation": metrics.get("policy_explanation", ""),
        "fwdtransforms": res.get("fwdtransforms", []),
        "invtransforms": res.get("invtransforms", [])
    }


def run_msd_task_benchmark(
    task_dir: str,
    pairs: List[Tuple[int, int]],
    reg_iterations: Optional[List[int]] = None,
    affine_iterations: Optional[List[int]] = None,
    output_json: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes benchmark evaluation across multiple pairs for an MSD task.

    Parameters:
    -----------
    task_dir : str
        Path to unpacked MSD task directory.
    pairs : list of tuple of (int, int)
        List of (fixed_idx, moving_idx) tuples to register.
    reg_iterations : list of int, optional
        Deformable iterations per pyramid level.
    affine_iterations : list of int, optional
        Affine iterations per pyramid level.
    output_json : str, optional
        Destination path to persist results JSON.

    Returns:
    --------
    list of dict
        List of pair evaluation records.
    """
    results = []
    print("=" * 80)
    print(f"AUTOMATED DECATHLON BENCHMARK: {os.path.basename(task_dir.rstrip('/'))}")
    print("=" * 80, flush=True)

    for p_idx, (f_idx, m_idx) in enumerate(pairs):
        print(f"\n[{p_idx + 1}/{len(pairs)}] Registering Case {m_idx:03d} -> Case {f_idx:03d}...", flush=True)
        res = evaluate_msd_pair(
            task_dir=task_dir,
            fixed_idx=f_idx,
            moving_idx=m_idx,
            reg_iterations=reg_iterations,
            affine_iterations=affine_iterations,
            verbose=False
        )
        results.append(res)
        print(
            f"  Result: Dice {res['initial_dice_sym']:.4f} -> {res['final_dice_sym']:.4f} "
            f"({res['dice_gain']:+.4f}) | Folds={res['folding_pct']:.4f}% | {res['time_seconds']:.1f}s | "
            f"Diagnosed: {res['diagnosed_fixed_anatomy']} ({res['diagnosed_fixed_modality']})",
            flush=True
        )

    # Print Summary Table
    d_initial = [r["initial_dice_sym"] for r in results]
    d_final = [r["final_dice_sym"] for r in results]
    d_gains = [r["dice_gain"] for r in results]
    folds = [r["folding_pct"] for r in results]
    times = [r["time_seconds"] for r in results]

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("-" * 80)
    print(f"Mean Initial DICE : {np.mean(d_initial):.4f}")
    print(f"Mean Final DICE   : {np.mean(d_final):.4f} (Mean Gain: {np.mean(d_gains):+.4f})")
    print(f"Mean Folding %    : {np.mean(folds):.4f}%")
    print(f"Mean Runtime      : {np.mean(times):.1f}s")
    print(f"Win Rate (>0 gain): {sum(g > 0 for g in d_gains)}/{len(d_gains)} ({np.mean([g > 0 for g in d_gains])*100:.1f}%)")
    print("=" * 80, flush=True)

    if output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark results to {output_json}")

    return results
