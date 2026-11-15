"""
syntx.benchmark.msd — ``syntx.auto_reg`` on Medical Segmentation Decathlon (MSD) pairs
=====================================================================================

For case pairs of an unpacked MSD task (a directory with ``dataset.json``), registers the
moving case to the fixed one with ``syntx.auto_reg`` and records the label Dice before and
after, folding, runtime, and the modality / anatomy diagnosis ``auto_reg`` reports. The
diagnosis is recorded, not checked against the task metadata.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import ants

import syntx
from syntx.provenance import with_provenance
from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.pyramid import cap_ants_image


STRATUM_A_TASKS = [
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task07_Pancreas",
    "Task09_Spleen",
]

STRATUM_B_TASKS = [
    "Task01_BrainTumour",
    "Task06_Lung",
    "Task08_HepaticVessel",
    "Task10_Colon",
]


def _prepare_image_and_label(
    img_path: str,
    lbl_path: Optional[str] = None,
    channel: Optional[int] = None,
    max_dimension: Optional[int] = 256,
    task_name: Optional[str] = None,
    combine_parenchyma: bool = True,
) -> Tuple[ants.ANTsImage, Optional[ants.ANTsImage]]:
    """Read an image and its label map.

    - 4-D image: defaults to channel 1 (T1w structural channel) for BrainTumour (Task01),
      and channel 0 for other tasks unless ``channel`` is explicitly specified.
    - 4-D label: volume 0 is used.
    - Task-specific semantics:
      - Task03 (Liver) and Task07 (Pancreas): combine parenchymal and tumor labels
        ((lbl == 1) | (lbl == 2)) when ``combine_parenchyma=True`` to avoid cancer carve-out penalties.
      - Task05 (Prostate): combine peripheral and transition zones ((lbl == 1) | (lbl == 2))
        for whole gland overlap.
    - Anisotropy-preserving capping:
      If ``max_dimension`` is specified and max(img.shape) > max_dimension, downsamples the
      working image using ``cap_ants_image(img, ..., policy='deformable')`` to preserve native
      thick-slice z spacing. Labels are NEVER downsampled, preserving native full resolution.
    """
    img = ants.image_read(img_path)
    lbl = ants.image_read(lbl_path) if (lbl_path and os.path.exists(lbl_path)) else None

    # Determine default channel for 4D images
    if img.dimension == 4:
        if channel is not None:
            ch_idx = int(channel)
        elif task_name is not None and ("BrainTumour" in task_name or "Task01" in task_name):
            ch_idx = 1  # T1w structural channel
        else:
            ch_idx = 0
        arr = img.numpy()[..., ch_idx]
        img = ants.from_numpy(arr, origin=img.origin[:3], spacing=img.spacing[:3], direction=img.direction[:3, :3])

    if lbl is not None and lbl.dimension == 4:
        arr_l = lbl.numpy()[..., 0]
        lbl = ants.from_numpy(arr_l, origin=lbl.origin[:3], spacing=lbl.spacing[:3], direction=lbl.direction[:3, :3])

    # Apply task-specific parenchymal label combinations
    if lbl is not None and combine_parenchyma and task_name is not None:
        arr_l = lbl.numpy()
        if any(k in task_name for k in ("Liver", "Task03", "Pancreas", "Task07", "Prostate", "Task05")):
            # Combine label 1 and label 2 for whole-organ / parenchymal coverage
            combined = np.where((arr_l == 1) | (arr_l == 2), 1.0, 0.0).astype(np.float32)
            lbl = ants.from_numpy(combined, origin=lbl.origin, spacing=lbl.spacing, direction=lbl.direction)

    # Anisotropy-preserving downsampling for working image (policy='deformable' preserves native z)
    if max_dimension is not None and max(img.shape) > max_dimension:
        scale = float(max_dimension / max(img.shape))
        max_res = float(min(img.spacing) / scale)
        img = cap_ants_image(img, max_resolution_mm=max_res, policy="deformable")

    return img, lbl


@with_provenance("syntx.benchmark.msd.evaluate_msd_pair")
def evaluate_msd_pair(
    task_dir: str,
    fixed_idx: int,
    moving_idx: int,
    reg_iterations: Optional[List[int]] = None,
    max_dimension: Optional[int] = 256,
    verbose: bool = False
) -> Dict[str, Any]:
    """Register MSD training case ``moving_idx`` to case ``fixed_idx`` with ``syntx.auto_reg``.

    Images and labels are read with ``_prepare_image_and_label``.
    Initial and final Dice are evaluated on native full-resolution labels and images with
    ``compute_bidirectional_dice`` using exact transform inversion (whichtoinvert_inv=[True, False]).

    Parameters
    ----------
    task_dir : str
        Unpacked MSD task directory containing ``dataset.json``.
    fixed_idx, moving_idx : int
        Indices into ``dataset.json['training']``.
    reg_iterations : list of int, optional
        Deformable iterations; default [40, 20, 10]. (The affine stage is ``auto_reg``'s
        ``robust_affine``; it has no iteration schedule here.)
    max_dimension : int or None, default 256
        Downsample working image when largest dimension exceeds this (using anisotropy-preserving
        cap_ants_image); labels remain at native full resolution.
    verbose : bool, default False
        Passed to ``auto_reg``.

    Returns
    -------
    dict
        'task_name', 'fixed_case', 'moving_case' (image file names), 'initial_dice_sym',
        'final_dice_sym', 'final_dice_fixed', 'final_dice_moving', 'dice_gain' (final - initial),
        'folding_pct', 'min_jac', 'time_seconds', 'diagnosed_relationship',
        'diagnosed_fixed_anatomy', 'diagnosed_fixed_modality', 'policy_explanation',
        'fwdtransforms', 'invtransforms', plus 'provenance' added by the ``with_provenance`` decorator.
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

    # Load native full-resolution images and labels
    fi_native, fl_native = _prepare_image_and_label(fix_img_p, fix_lbl_p, max_dimension=None, task_name=task_name)
    mi_native, ml_native = _prepare_image_and_label(mov_img_p, mov_lbl_p, max_dimension=None, task_name=task_name)

    # Working images for registration (anisotropy-preserving capping if requested)
    if max_dimension is not None and (max(fi_native.shape) > max_dimension or max(mi_native.shape) > max_dimension):
        fi, _ = _prepare_image_and_label(fix_img_p, None, max_dimension=max_dimension, task_name=task_name)
        mi, _ = _prepare_image_and_label(mov_img_p, None, max_dimension=max_dimension, task_name=task_name)
    else:
        fi, mi = fi_native, mi_native

    # Initial overlap on native full-resolution images (unaligned baseline)
    d_fix_0, d_mov_0, d_sym_0 = compute_bidirectional_dice(fl_native, ml_native, fi_native, mi_native, [], [], [])

    # Run auto_reg with autonomous diagnosis & policy synthesis
    t0 = time.time()
    res = syntx.auto_reg(
        fixed=fi,
        moving=mi,
        reg_iterations=reg_iterations or [40, 20, 10],
        verbose=verbose
    )
    t_elapsed = time.time() - t0

    # Evaluate final composite transforms on native full-resolution labels and images
    fwd_tx = res.get("fwdtransforms", [])
    inv_tx = res.get("invtransforms", [])
    n_inv = len(inv_tx)
    whichtoinvert_inv = [True] if n_inv == 1 else ([True, False] if n_inv == 2 else [True] + [False] * (n_inv - 1)) if n_inv > 0 else []

    d_fix_final, d_mov_final, d_sym_final = compute_bidirectional_dice(
        fl_native, ml_native, fi_native, mi_native,
        fwd_tx, inv_tx, whichtoinvert_inv
    )

    metrics = res.get("metrics", {})
    folding_pct = metrics.get("folding_pct", float("nan"))
    min_jac = metrics.get("jac_min", float("nan"))
    diag = metrics.get("diagnosis", {})

    return {
        "task_name": task_name,
        "fixed_case": os.path.basename(fix_img_p),
        "moving_case": os.path.basename(mov_img_p),
        "initial_dice_sym": float(d_sym_0),
        "final_dice_sym": float(d_sym_final),
        "final_dice_fixed": float(d_fix_final),
        "final_dice_moving": float(d_mov_final),
        "dice_gain": float(d_sym_final - d_sym_0),
        "folding_pct": float(folding_pct),
        "min_jac": float(min_jac),
        "time_seconds": float(t_elapsed),
        "diagnosed_relationship": diag.get("relationship", "UNKNOWN"),
        "diagnosed_fixed_anatomy": diag.get("fixed_anatomy", "UNKNOWN"),
        "diagnosed_fixed_modality": diag.get("fixed_modality", "UNKNOWN"),
        "policy_explanation": metrics.get("policy_explanation", ""),
        "fwdtransforms": fwd_tx,
        "invtransforms": inv_tx,
    }


def run_msd_task_benchmark(
    task_dir: str,
    pairs: List[Tuple[int, int]],
    reg_iterations: Optional[List[int]] = None,
    max_dimension: Optional[int] = 256,
    output_json: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run ``evaluate_msd_pair`` on each (fixed_idx, moving_idx) pair and print a summary.

    The summary gives the mean initial / final Dice, mean gain, mean folding %, mean runtime
    (NaN values skipped) and the fraction of pairs with a positive gain.

    Parameters
    ----------
    task_dir : str
        Unpacked MSD task directory.
    pairs : list of (int, int)
        (fixed_idx, moving_idx) tuples.
    reg_iterations : list of int, optional
        Passed to ``evaluate_msd_pair`` (default [40, 20, 10] there).
    max_dimension : int or None, default 256
        Passed to ``evaluate_msd_pair``.
    output_json : str, optional
        If given, the list of records is written there as JSON (directory created).

    Returns
    -------
    list of dict
        The ``evaluate_msd_pair`` records, in input order.
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
            max_dimension=max_dimension,
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
    print(f"Mean Initial DICE : {np.nanmean(d_initial):.4f}")
    print(f"Mean Final DICE   : {np.nanmean(d_final):.4f} (Mean Gain: {np.nanmean(d_gains):+.4f})")
    print(f"Mean Folding %    : {np.nanmean(folds):.4f}%")
    print(f"Mean Runtime      : {np.nanmean(times):.1f}s")
    print(f"Win Rate (>0 gain): {sum(g > 0 for g in d_gains)}/{len(d_gains)} ({np.mean([g > 0 for g in d_gains])*100:.1f}%)")
    print("=" * 80, flush=True)

    if output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark results to {output_json}")

    return results
